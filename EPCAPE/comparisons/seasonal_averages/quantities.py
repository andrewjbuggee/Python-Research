"""The quantity behind each spreadsheet row, computed from the downloaded data.

Each function returns the samples whose seasonal mean ± std goes into the
table (a time-indexed pandas Series or DataFrame) plus, where useful, a small
dict describing what was kept. Units are in the names (``_m``, ``_gm2``,
``_mm_h``, ``_ugm3``).

Quality control
---------------
Every ARM variable that has a ``qc_<var>`` companion is screened with
``EPCAPE.analysis_tools.qc.qc_is_zero``: a sample is kept only if its QC value is exactly 0,
i.e. it passed every test, including those ARM assesses as "Indeterminate".
Variables without a QC field are used as distributed; the notebook prints
which ones those are (``qc_report``).

Time averaging
--------------
Continuous series are averaged over a fixed window before statistics, so
instruments with 4-s and 1-min sampling are compared on the same footing;
Kavin's SW transmittance note specifies 5-min averages. The SW and LW
functions take the window in minutes (``window_min``, any non-negative
length; 0 = no averaging, every QC = 0 sample counts); the others take a
pandas period string (default "5min"). The standard deviation of window
means is smaller than that of the raw samples, because variability shorter
than the window is averaged out.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from EPCAPE.analysis_tools import qc, units


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def qc0_values(ds: xr.Dataset, var: str) -> Tuple[xr.DataArray, Dict[str, object]]:
    """`var` with every sample whose qc_<var> is not exactly 0 set to NaN.

    Returns (masked values, info). info: has_qc, n (all samples), n_finite
    (finite before QC), n_kept (finite after QC)."""
    da = ds[var]
    ok = qc.qc_is_zero(ds, var)
    out = da.where(ok)
    info = {
        "variable": var,
        "has_qc": qc.has_qc(ds, var),
        "n": int(da.size),
        "n_finite": int(np.isfinite(da.values).sum()),
        "n_kept": int(np.isfinite(out.values).sum()),
    }
    return out, info


def qc_report(infos: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Table of how many samples QC = 0 kept, per variable."""
    df = pd.DataFrame(list(infos), columns=["variable", "has_qc", "n", "n_finite", "n_kept"])
    df["kept_pct_of_finite"] = (100.0 * df["n_kept"] / df["n_finite"].where(df["n_finite"] > 0)).round(1)
    return df.set_index("variable")


def to_series(da: xr.DataArray, name: Optional[str] = None) -> pd.Series:
    """1-D time DataArray -> pandas Series indexed by time."""
    s = da.to_series()
    s.name = name or da.name
    return s


def grid_mean(series: pd.Series, period: str = "5min", min_count: int = 1) -> pd.Series:
    """Mean of the finite samples in each `period` bin; bins with fewer than
    `min_count` finite samples are NaN (not 0)."""
    g = series.resample(period)
    mean = g.mean()
    return mean.where(g.count() >= min_count)


def window_mean(series: pd.Series, window_min: float, min_count: int = 1) -> pd.Series:
    """Average `series` over consecutive, non-overlapping windows `window_min` minutes long.

    window_min is any non-negative length in minutes, fractional allowed
    (0.5, 1, 5, 7.5, 60, 1440, ...).

    * window_min > 0: mean of the finite samples in each window, labelled by
      the window's start time. Windows are anchored at midnight UTC of the
      first day. Windows with fewer than `min_count` finite samples are NaN,
      not 0. A window shorter than the sampling interval leaves the data
      unchanged apart from empty windows, which are NaN.
    * window_min == 0: no further averaging. The finite samples are returned
      as they are, so the seasonal statistics are taken over every sample that
      passed QC, at the resolution the product is distributed at. For RADFLUX
      that is 1-min means (each the mean of 60 one-second SKYRAD samples;
      epcskyrad60s: sampling_interval 1 s, averaging_interval 60 s), so 0 and
      1 give the same result.
    """
    if not np.isfinite(window_min) or window_min < 0:
        raise ValueError(f"window_min must be a finite number >= 0 (minutes), got {window_min!r}")
    if window_min == 0:
        return series[np.isfinite(series)]
    return grid_mean(series, pd.Timedelta(minutes=float(window_min)), min_count=min_count)


def describe_window(window_min: float) -> str:
    """Words for one averaged sample, for table labels: '5-min mean', 'every QC = 0 sample'."""
    if window_min == 0:
        return "every QC = 0 sample (no further averaging)"
    return f"{window_min:g}-min mean"


# ---------------------------------------------------------------------------
# radiation (RADFLUX; Long & Ackerman 2000, JGR 105, 15609; Long & Turner 2008, JGR 113, D18206)
# ---------------------------------------------------------------------------
def sw_transmittance(
    ds: xr.Dataset, *, sza_max_deg: float = 80.0, window_min: float = 5.0
) -> Tuple[pd.Series, Dict[str, object]]:
    """SW transmittance = <SW down> / <clear-sky SW down>, each averaged over `window_min` minutes.

    window_min: averaging window in minutes, any non-negative length (see
    ``window_mean``). With window_min = 0 there is no further averaging: the
    ratio is taken for every RADFLUX 1-min mean that passes QC and the daytime
    cut, and the seasonal statistics are over those ratios. RADFLUX values
    are 1-min means of 1-s SKYRAD samples, timestamped at the END of the
    minute (time_bounds = [-60 s, 0]). That is the mean of instantaneous ratios, which is not the same
    as the ratio of seasonal-mean irradiances.

    Kavin's note: "ratio of five-minute-averaged downwelling shortwave
    irradiances from SKYRAD (radflux1long dataset) to an idealized clear sky
    analytical model calculation (Atwater and Ball, 1981)". RADFLUX's
    ``downwelling_shortwave`` is the SKYRAD measurement. The denominator here is
    RADFLUX's own ``clearsky_downwelling_shortwave``, an empirical clear-sky
    fit to this site's clear periods (Long & Ackerman 2000), not the Atwater &
    Ball analytical model, whose equations I could not verify from an
    accessible source. Clear-sky models differ by a few percent, which moves
    the ratio by about as much.

    Only samples with the sun higher than 90 - `sza_max_deg` degrees are used:
    near the horizon both irradiances are small and the ratio is noisy. Both
    numerator and denominator are averaged over the same valid samples so the
    ratio is not biased by unequal gaps. Values > 1 (cloud-edge enhancement)
    are kept."""
    sw, info_sw = qc0_values(ds, "downwelling_shortwave")
    cs, info_cs = qc0_values(ds, "clearsky_downwelling_shortwave")
    mu0 = ds["cosine_zenith"]
    day = mu0 > np.cos(np.deg2rad(sza_max_deg))
    # The same samples go into numerator and denominator: both QC = 0 (the clear-sky
    # estimate has no QC field), clear-sky > 0, and the sun above the cut.
    both = np.isfinite(sw) & np.isfinite(cs) & (cs > 0) & day
    sw_avg = window_mean(to_series(sw.where(both)), window_min)
    cs_avg = window_mean(to_series(cs.where(both)), window_min)
    ratio = (sw_avg / cs_avg).where(cs_avg > 0).rename("sw_transmittance")
    info = {
        "qc": [info_sw, info_cs],
        "sza_max_deg": sza_max_deg,
        "window_min": window_min,
        "n_values": int(ratio.notna().sum()),
    }
    return ratio, info


SOLAR_CONSTANT_WM2 = 1361.0  # total solar irradiance at 1 AU (Kopp & Lean 2011, GRL 38, L01706)


def sw_clearness_index(
    ds: xr.Dataset, *, sza_max_deg: float = 80.0, window_min: float = 5.0
) -> Tuple[pd.Series, Dict[str, object]]:
    """<SW down> / <top-of-atmosphere SW down>, each averaged over `window_min` minutes
    (the clearness index). window_min = 0: sample-by-sample ratios, no averaging.

    TOA irradiance on a horizontal surface = S0 (1 + 0.033 cos(2 pi n / 365)) mu0,
    with n the day of year; the cosine term approximates the Earth-Sun distance
    (Duffie & Beckman, Solar Engineering of Thermal Processes, Eq. 1.4.1a).
    Not the sheet's definition: a sensitivity test bracketing how bright the
    clear-sky denominator would have to be to give the sheet's values."""
    sw, info_sw = qc0_values(ds, "downwelling_shortwave")
    mu0 = ds["cosine_zenith"]
    doy = pd.DatetimeIndex(ds["time"].values).dayofyear.values
    toa = SOLAR_CONSTANT_WM2 * (1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)) * mu0
    ok = np.isfinite(sw) & (mu0 > np.cos(np.deg2rad(sza_max_deg)))
    sw_avg = window_mean(to_series(sw.where(ok)), window_min)
    toa_avg = window_mean(to_series(toa.where(ok)), window_min)
    ratio = (sw_avg / toa_avg).where(toa_avg > 0).rename("clearness_index")
    return ratio, {"qc": [info_sw], "window_min": window_min}


def lw_down(ds: xr.Dataset, *, window_min: float = 5.0) -> Tuple[pd.Series, Dict[str, object]]:
    """Downwelling LW irradiance (W m-2), QC = 0, day and night, averaged over `window_min` minutes.

    window_min: averaging window in minutes, any non-negative length (see
    ``window_mean``); 0 = no further averaging, i.e. the statistics are over
    every RADFLUX 1-min mean (60 one-second samples each) that passes QC."""
    lw, info = qc0_values(ds, "downwelling_longwave")
    out = window_mean(to_series(lw), window_min).rename("lw_down_wm2")
    return out, {"qc": [info], "window_min": window_min}


# ---------------------------------------------------------------------------
# liquid water path
# ---------------------------------------------------------------------------
def lwp_gm2(ds: xr.Dataset, var: str, *, period: str = "5min") -> Tuple[pd.Series, Dict[str, object]]:
    """Liquid water path in g m-2 (unit read from the file), QC = 0, averaged to `period`.

    All QC = 0 samples are kept, including clear sky (LWP ~ 0) and small
    negative values within the retrieval noise, because a campaign-mean LWP
    is an all-sky mean. Use ``cloudy_only`` for a cloudy-sky mean."""
    values, info = qc0_values(ds, var)
    values = units.water_path_to_gm2(values)
    return grid_mean(to_series(values), period).rename(f"{var}_gm2"), {"qc": [info]}


def near_rain(
    rate_mm_h: pd.Series, index: pd.DatetimeIndex, *, window_min: float = 30.0, period: str = "5min"
) -> np.ndarray:
    """True for the times in `index` within ±`window_min` of any rain (rate > 0) in `rate_mm_h`.

    Used to drop microwave-radiometer LWP while liquid may be on the radome,
    which biases retrieved LWP high (rain drops and a wet window emit at 23.8
    and 31.4 GHz). Rate samples that are NaN (e.g. failed QC) and times the
    rain record does not cover count as dry."""
    wet = (rate_mm_h > 0).astype(float).resample(period).max()
    n_bins = int(round(2.0 * window_min / (pd.Timedelta(period).total_seconds() / 60.0))) + 1
    wet = wet.rolling(n_bins, center=True, min_periods=1).max()
    return wet.reindex(index).fillna(0.0).to_numpy() > 0


def cloudy_only(series: pd.Series, threshold: float) -> pd.Series:
    """Keep samples >= threshold (e.g. LWP >= 20 g m-2 as 'cloudy')."""
    return series.where(series >= threshold)


# ---------------------------------------------------------------------------
# cloud boundaries
# ---------------------------------------------------------------------------
def lowest_layer_top_m(
    ds: xr.Dataset, *, max_top_m: float = 3000.0, single_layer: bool = False, period: str = "5min"
) -> Tuple[pd.Series, Dict[str, object]]:
    """Top (m above ground) of the lowest KAZR-ARSCL hydrometeor layer, for low clouds.

    ``cloud_layer_top_height[:, 0]`` is the top of the lowest layer (layers
    are ordered upward). A sample counts when that layer exists and its top
    is <= `max_top_m` (3 km, the "low cloud" bound used for the EPCAPE Low
    Cloud Periods). With single_layer=True, samples with a second layer are
    dropped. Empty layer slots hold the flag value -1 ("clear_sky"), not NaN. ARSCL distributes no QC field for these variables. Heights are
    above ground; the Scripps Pier site is ~7 m above sea level, which is
    negligible here.

    Note: ARSCL layers are hydrometeor layers, so drizzle below cloud base
    can lower the 'base', but the top is the cloud top in either case."""
    top = ds["cloud_layer_top_height"]
    base = ds["cloud_layer_base_height"]
    layer_dim = [d for d in top.dims if d != "time"][0]
    top0 = top.isel({layer_dim: 0})

    def present(x):
        # ARSCL writes -1 ("clear_sky" in flag_values) for an empty layer slot, not NaN.
        return np.isfinite(x) & (x >= 0)

    ok = present(top0) & present(base.isel({layer_dim: 0})) & (top0 > 0) & (top0 <= max_top_m)
    if single_layer:
        ok &= ~present(base.isel({layer_dim: 1}))
    s = to_series(top0.where(ok), "cloud_top_m")
    info = {
        "qc": [{"variable": "cloud_layer_top_height", "has_qc": False, "n": int(top0.size),
                "n_finite": int(present(top0).values.sum()), "n_kept": int(ok.values.sum())}],
        "max_top_m": max_top_m,
        "single_layer": single_layer,
    }
    return grid_mean(s, period), info


def ceilometer_cbh_m(
    ds: xr.Dataset, *, max_cbh_m: float = 3000.0, period: str = "5min"
) -> Tuple[pd.Series, Dict[str, object]]:
    """Lowest ceilometer cloud base (m above the instrument), QC = 0, low clouds only.

    A sample counts when detection_status is 1, 2 or 3 (one, two or three
    cloud bases detected; 4 = full obscuration reports vertical visibility
    instead, 0 = no significant backscatter) and first_cbh <= `max_cbh_m`.
    Heights are above the instrument, not sea level (ARM ceilometer handbook)."""
    cbh, info = qc0_values(ds, "first_cbh")
    status = ds["detection_status"]
    ok = np.isin(status.values, [1, 2, 3]) & np.isfinite(cbh.values) & (cbh.values <= max_cbh_m)
    s = to_series(cbh.where(ok), "cbh_m")
    return grid_mean(s, period), {"qc": [info], "max_cbh_m": max_cbh_m}


# ---------------------------------------------------------------------------
# boundary-layer height
# ---------------------------------------------------------------------------
def pblh_thermo_m(ds: xr.Dataset) -> Tuple[pd.Series, Dict[str, object]]:
    """PBLHTTHERMO best-estimate PBL height (m above ground), native 10-min samples.

    The file gives km above ground; there is no QC field. QC = 0 is applied
    anyway in case a later version adds one."""
    pbl, info = qc0_values(ds, "pbl_height")
    factor = {"km": 1000.0, "m": 1.0}[str(ds["pbl_height"].attrs.get("units", "km")).strip()]
    return to_series(pbl * factor, "pblh_thermo_m"), {"qc": [info]}


SONDE_PBL_VARS = (
    "pbl_height_liu_liang",
    "pbl_height_heffter",
    "pbl_height_bulk_richardson_pt25",
    "pbl_height_bulk_richardson_pt5",
)


def _scalar_qc0(ds: xr.Dataset, var: str) -> Tuple[float, float]:
    """A per-launch scalar: (value with QC = 0 applied, NaN otherwise; value before QC)."""
    if var not in ds:
        return np.nan, np.nan
    value = float(np.asarray(ds[var].values, dtype=float).ravel()[0])
    if f"qc_{var}" in ds:
        q = float(np.asarray(ds[f"qc_{var}"].values, dtype=float).ravel()[0])
        if not (np.isfinite(q) and q == 0):
            return np.nan, value
    return value, value


def sonde_launch(ds: xr.Dataset, *, surface_layer_hpa: float = 2.0) -> Dict[str, float]:
    """Reducer for one PBLHTSONDE launch file (use with sources.read_per_launch).

    Returns
      <method>_m_agl  PBL heights, QC = 0, converted from m above sea level
                      to m above ground by subtracting the site altitude `alt`
      <method>_raw_m_agl  the same before QC (to count what QC = 0 removed)
      p_sfc_hpa, t_sfc_c, rh_sfc_pct
                      medians of the samples within `surface_layer_hpa` of the
                      first valid pressure (the lowest ~15 m), used for the LCL.
                      These profile variables carry no QC field in this VAP.
    """
    alt = float(np.asarray(ds["alt"].values, dtype=float).ravel()[0]) if "alt" in ds else 0.0
    row = {}
    for v in SONDE_PBL_VARS:
        kept, raw = _scalar_qc0(ds, v)
        method = v.replace("pbl_height_", "")
        row[f"{method}_m_agl"] = kept - alt
        row[f"{method}_raw_m_agl"] = raw - alt
    p = np.asarray(ds["atm_pres"].values, dtype=float)
    t = np.asarray(ds["air_temp"].values, dtype=float)
    rh = np.asarray(ds["rh"].values, dtype=float)
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(rh) & (p > 0)
    if ok.any():
        p0 = p[ok][0]
        near = ok & (p >= p0 - surface_layer_hpa)
        row.update(
            p_sfc_hpa=float(np.median(p[near])),
            t_sfc_c=float(np.median(t[near])),
            rh_sfc_pct=float(np.median(rh[near])),
            n_sfc=int(near.sum()),
        )
    else:
        row.update(p_sfc_hpa=np.nan, t_sfc_c=np.nan, rh_sfc_pct=np.nan, n_sfc=0)
    row["alt_m"] = alt
    return row


def sondeparam_launch(ds: xr.Dataset) -> Dict[str, float]:
    """Reducer for one SONDEPARAM launch file: surface-parcel (and mixed-layer-parcel) LCL (m).

    ``lcl`` (km) is stored for every sample of the launch and three parcel
    types (1 = surface, 2 = most unstable, 3 = mixed layer); the value is
    the same along time, so the first finite surface-parcel value is used.
    The file has no qc_lcl; its ``data_quality`` bit mask (0 = all input
    checks passed) is required to be 0 for every sample of the launch,
    the QC = 0 equivalent for this product. The VAP documentation does not
    say whether lcl is above ground or sea level; the site is ~7 m above sea
    level, so the difference is negligible."""
    ptype = np.asarray(ds["parcel_type"].values).astype(int)
    lcl_all = np.atleast_2d(np.asarray(ds["lcl"].values, dtype=float))
    dq = np.asarray(ds["data_quality"].values, dtype=float) if "data_quality" in ds else np.array([0.0])
    dq_ok = bool(np.all(np.isfinite(dq)) and np.all(dq == 0))

    def first_finite_m(code: int) -> float:
        """LCL (m) of parcel type `code` (1 surface, 2 most unstable, 3 mixed layer)."""
        if not (ptype == code).any():
            return np.nan
        col = lcl_all[:, int(np.where(ptype == code)[0][0])]
        col = col[np.isfinite(col)]
        return float(col[0]) * 1000.0 if col.size else np.nan

    surface, mixed = first_finite_m(1), first_finite_m(3)
    return {
        "lcl_sondeparam_m": surface if dq_ok else np.nan,  # surface parcel, data_quality = 0
        "lcl_raw_m": surface,  # surface parcel before the data_quality screen
        "lcl_mixed_layer_m": mixed if dq_ok else np.nan,  # mixed-layer parcel (sensitivity)
        "data_quality_ok": dq_ok,
    }


# ---------------------------------------------------------------------------
# lifting condensation level
# ---------------------------------------------------------------------------
# Constants from Romps (2017), "Exact expression for the lifting condensation
# level", J. Atmos. Sci. 74, 3891-3900, doi:10.1175/JAS-D-17-0102.1 (values as
# in the paper's reference implementation; moderately confident they match
# the paper digit for digit, and they agree with standard values).
_TTRIP_K = 273.16  # triple-point temperature
_PTRIP_PA = 611.65  # triple-point vapour pressure
_E0V_JKG = 2.3740e6  # internal energy difference vapour - liquid at the triple point
_G_MS2 = 9.81
_RA = 287.04  # dry-air gas constant, J kg-1 K-1
_RV = 461.0  # water-vapour gas constant
_CVA, _CVV, _CVL = 719.0, 1418.0, 4119.0  # heat capacities at constant volume: dry air, vapour, liquid
_CPA, _CPV = _CVA + _RA, _CVV + _RV


def _lambertw_m1(x: np.ndarray) -> np.ndarray:
    """Lower branch W_{-1}(x) of the Lambert W function for -1/e <= x < 0.

    Halley iteration on w e^w = x (Corless et al. 1996, Adv. Comput. Math.
    5, 329). Implemented here because scipy is not in this project's
    environment. Starting guesses: the branch-point series
    w ~ -1 - sqrt(2 (1 + e x)) near x = -1/e, and the asymptotic
    w ~ ln(-x) - ln(-ln(-x)) near x = 0-."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    valid = (x >= -1.0 / np.e) & (x < 0)
    xv = np.maximum(x[valid], -1.0 / np.e)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(
            xv < -0.25,
            -1.0 - np.sqrt(np.maximum(2.0 * (1.0 + np.e * xv), 0.0)),
            np.log(-xv) - np.log(-np.log(-xv)),
        )
        for _ in range(60):
            ew = np.exp(w)
            f = w * ew - xv
            wp1 = w + 1.0
            denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1)
            step = np.where(np.abs(wp1) > 1e-12, f / denom, 0.0)
            w = w - step
            if np.all(np.abs(step) <= 1e-13 * np.abs(w)):
                break
    out[valid] = w
    return out


def saturation_vapour_pressure_liquid_pa(t_k):
    """Saturation vapour pressure over liquid (Pa), Romps (2017) form:
    p_v* = p_trip (T/T_trip)^((c_pv - c_vl)/R_v) exp[(E0v - (c_vv - c_vl) T_trip)/R_v (1/T_trip - 1/T)]."""
    t_k = np.asarray(t_k, dtype=float)
    return (
        _PTRIP_PA
        * (t_k / _TTRIP_K) ** ((_CPV - _CVL) / _RV)
        * np.exp((_E0V_JKG - (_CVV - _CVL) * _TTRIP_K) / _RV * (1.0 / _TTRIP_K - 1.0 / t_k))
    )


def lcl_romps2017_m(p_pa, t_k, rh_frac):
    """Height of the LCL above the parcel's starting level (m), exact for liquid (Romps 2017).

    A parcel lifted dry-adiabatically cools as T(z) = T - g z / c_pm and its
    vapour pressure falls with pressure, p_v(z) = p_v (T(z)/T)^(c_pm/R_m).
    Setting p_v(z) equal to p_v*(T(z)) and solving gives

        T_LCL = c T / W_{-1}( RH^(1/a) c e^c ),   c = b / a,
        a = c_pm/R_m + (c_vl - c_pv)/R_v,
        b = -(E0v - (c_vv - c_vl) T_trip) / (R_v T),
        z_LCL = (c_pm / g) (T - T_LCL),

    where c_pm and R_m are the moist-air heat capacity and gas constant from
    the specific humidity q_v. RH is with respect to liquid water (what
    radiosondes report), as a fraction; RH >= 1 gives z_LCL = 0.

    Parameters: p_pa pressure (Pa), t_k temperature (K), rh_frac in [0, 1+].
    """
    p = np.asarray(p_pa, dtype=float)
    t = np.asarray(t_k, dtype=float)
    rh = np.clip(np.asarray(rh_frac, dtype=float), 1e-6, 1.0)
    pv = rh * saturation_vapour_pressure_liquid_pa(t)
    qv = _RA * pv / (_RV * p + (_RA - _RV) * pv)  # specific humidity
    rm = (1.0 - qv) * _RA + qv * _RV
    cpm = (1.0 - qv) * _CPA + qv * _CPV
    a = cpm / rm + (_CVL - _CPV) / _RV
    b = -(_E0V_JKG - (_CVV - _CVL) * _TTRIP_K) / (_RV * t)
    c = b / a
    t_lcl = c * t / _lambertw_m1(rh ** (1.0 / a) * c * np.exp(c))
    z = cpm / _G_MS2 * (t - t_lcl)
    return np.where(rh >= 1.0, 0.0, z)


def dewpoint_magnus_c(t_c, rh_pct):
    """Dew point (degC) from T (degC) and RH (%) with the Magnus form, a = 17.625,
    b = 243.04 degC (Alduchov & Eskridge 1996, J. Appl. Meteor. 35, 601; Lawrence 2005)."""
    t_c = np.asarray(t_c, dtype=float)
    gamma = np.log(np.asarray(rh_pct, dtype=float) / 100.0) + 17.625 * t_c / (243.04 + t_c)
    return 243.04 * gamma / (17.625 - gamma)


def lcl_lawrence2005_m(t_c, td_c):
    """LCL height (m) from the rule of thumb z ~ 125 m K-1 (T - Td)
    (Lawrence 2005, BAMS 86, 225-233). A cross-check on lcl_romps2017_m."""
    return 125.0 * (np.asarray(t_c, dtype=float) - np.asarray(td_c, dtype=float))


# ---------------------------------------------------------------------------
# precipitation
# ---------------------------------------------------------------------------
def _step_hours(rate: pd.Series) -> float:
    """Sampling interval of a rain-rate series, in hours (median spacing)."""
    return float(pd.Series(rate.index).diff().median() / pd.Timedelta(hours=1))


def rain_events(
    rate_mm_h: pd.Series,
    *,
    threshold_mm_h: float = 0.1,
    max_gap_min: float = 60.0,
    min_accum_mm: float = 0.1,
) -> pd.DataFrame:
    """Rain events from a rain-rate series (mm h-1, ~1-min samples).

    A sample is 'raining' when rate >= `threshold_mm_h`. Raining samples
    separated by <= `max_gap_min` belong to one event. For each event:
        start, end           first raining sample, last raining sample + one interval
        duration_h           end - start
        accumulation_mm      sum of rate x interval over the event's raining samples
        mean_rate_mm_h       accumulation_mm / duration_h (event-mean intensity)
        peak_rate_mm_h
    Events with accumulation < `min_accum_mm` are dropped. NaN (QC-failed)
    samples count as not raining. The index is the event start time."""
    rate = rate_mm_h.astype(float)
    dt_h = _step_hours(rate)
    wet = rate.where(rate >= threshold_mm_h).dropna()
    cols = ["start", "end", "duration_h", "accumulation_mm", "mean_rate_mm_h", "peak_rate_mm_h", "n_samples"]
    if wet.empty:
        return pd.DataFrame(columns=cols).set_index("start")
    gap_min = pd.Series(wet.index).diff().dt.total_seconds().div(60.0).values
    event_id = np.cumsum(np.r_[True, gap_min[1:] > max_gap_min])
    rows = []
    for _, chunk in wet.groupby(event_id):
        start, end = chunk.index[0], chunk.index[-1] + pd.Timedelta(hours=dt_h)
        duration_h = (end - start) / pd.Timedelta(hours=1)
        accum = float(chunk.sum() * dt_h)
        rows.append(
            {
                "start": start,
                "end": end,
                "duration_h": duration_h,
                "accumulation_mm": accum,
                "mean_rate_mm_h": accum / duration_h,
                "peak_rate_mm_h": float(chunk.max()),
                "n_samples": int(chunk.size),
            }
        )
    events = pd.DataFrame(rows).set_index("start")
    return events[events["accumulation_mm"] >= min_accum_mm]


def daily_rain_mm(rate_mm_h: pd.Series, *, min_day_mm: float = 0.1, min_coverage: float = 0.8) -> pd.Series:
    """Daily (UTC) rain accumulation (mm) on rain days (>= `min_day_mm`).

    Days where fewer than `min_coverage` of the expected samples are valid
    are dropped, so an outage does not look like a dry or light day."""
    rate = rate_mm_h.astype(float)
    dt_h = _step_hours(rate)
    expected = 24.0 / dt_h
    total = (rate * dt_h).resample("1D").sum(min_count=1)
    coverage = rate.resample("1D").count() / expected
    total = total.where(coverage >= min_coverage)
    return total.where(total >= min_day_mm).dropna().rename("daily_rain_mm")


def hourly_rain_mm(rate_mm_h: pd.Series, *, min_hour_mm: float = 0.1, min_coverage: float = 0.8) -> pd.Series:
    """Hourly (UTC) rain accumulation (mm, i.e. mm per hour) in wet hours (>= `min_hour_mm`).

    One literal reading of the table's "Rain Accumulation ... mm/hr". Hours
    with fewer than `min_coverage` of the expected samples valid are dropped."""
    rate = rate_mm_h.astype(float)
    dt_h = _step_hours(rate)
    total = (rate * dt_h).resample("1h").sum(min_count=1)
    coverage = rate.resample("1h").count() * dt_h
    total = total.where(coverage >= min_coverage)
    return total.where(total >= min_hour_mm).dropna().rename("hourly_rain_mm")


def raining_rates(rate_mm_h: pd.Series, *, threshold_mm_h: float = 0.1) -> pd.Series:
    """Rain-rate samples while raining (rate >= threshold), mm h-1."""
    return rate_mm_h.where(rate_mm_h >= threshold_mm_h).dropna()


# ---------------------------------------------------------------------------
# GCVI cloud/haze residuals measured by the AMS (Mt. Soledad)
# ---------------------------------------------------------------------------
def gcvi_segments(gcvi: pd.DataFrame, *, ef_min: float = 1.0, max_gap_s: float = 60.0) -> pd.DataFrame:
    """Continuous GCVI sampling periods from the 1-s enhancement-factor record.

    The GCVI is taken to be sampling droplets when its enhancement factor is
    finite and > `ef_min` (EF <= 0 or inf marks start-up/idle states in the
    file; when sampling, EF ~ 5). Samples separated by more than `max_gap_s`
    start a new segment. Columns: start, end, duration_min, mean_EF,
    median_cut_um."""
    on = gcvi["EF"].where(np.isfinite(gcvi["EF"]) & (gcvi["EF"] > ef_min)).dropna()
    gap_s = pd.Series(on.index).diff().dt.total_seconds().values
    seg = np.cumsum(np.r_[True, gap_s[1:] > max_gap_s])
    g = pd.DataFrame({"EF": on.values, "cut_um": gcvi["cut_um"].reindex(on.index).values, "seg": seg}, index=on.index)
    out = g.groupby("seg").agg(
        start=("EF", lambda s: s.index[0]),
        end=("EF", lambda s: s.index[-1] + pd.Timedelta(seconds=1)),
        mean_EF=("EF", "mean"),
        median_cut_um=("cut_um", "median"),
    )
    out["duration_min"] = (out["end"] - out["start"]).dt.total_seconds() / 60.0
    return out.reset_index(drop=True)


def gcvi_ams_residuals(
    ams: pd.DataFrame,
    gcvi: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    min_duration_min: float = 15.0,
    vmode_min: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """AMS samples taken behind the GCVI, with the enhancement factor removed.

    An AMS sample (timestamp = start of its `vmode_min` V-mode period, per the
    file README) counts when that whole period lies inside a GCVI segment of
    at least `min_duration_min` ("minimum sampling 15 min" in the table).
    Behind a CVI, concentrations are enhanced by the EF (the ratio of the
    air volume the droplets were drawn from to the carrier flow; Shingler
    et al. 2012, AMT 5, 1259, Eq. 2), so the ambient-equivalent residual
    concentration is measured / EF, with EF averaged over the sample's
    V-mode period from the 1-s record.

    Returns (samples, used_segments):
      samples        per AMS sample: <species>_ugm3 as measured, <species>_resid_ugm3
                     (/ EF), EF, segment
      used_segments  the >= min_duration_min segments that contain >= 1 AMS sample
    """
    long_segs = segments[segments["duration_min"] >= min_duration_min].reset_index(drop=True)
    t_ams = ams.index.values.astype("datetime64[ns]")
    t_end = t_ams + np.timedelta64(int(vmode_min * 60), "s")
    starts = long_segs["start"].values.astype("datetime64[ns]")
    ends = long_segs["end"].values.astype("datetime64[ns]")
    # segment whose start is the last one at or before the AMS sample start
    k = np.searchsorted(starts, t_ams, side="right") - 1
    inside = (k >= 0) & (t_end <= ends[np.clip(k, 0, None)])
    # mean EF over [t, t + vmode) from cumulative sums of the 1-s record
    ef = gcvi["EF"].where(np.isfinite(gcvi["EF"]) & (gcvi["EF"] > 0))
    t1 = ef.index.values.astype("datetime64[ns]")
    finite = np.isfinite(ef.values)
    csum = np.r_[0.0, np.cumsum(np.where(finite, ef.values, 0.0))]
    ccnt = np.r_[0, np.cumsum(finite)]
    i0 = np.searchsorted(t1, t_ams[inside], side="left")
    i1 = np.searchsorted(t1, t_end[inside], side="left")
    n = ccnt[i1] - ccnt[i0]
    ef_mean = np.where(n > 0, (csum[i1] - csum[i0]) / np.maximum(n, 1), np.nan)

    samples = ams[inside].copy()
    samples["EF"] = ef_mean
    samples["segment"] = k[inside]
    for col in [c for c in ams.columns if c.endswith("_ugm3")]:
        samples[col.replace("_ugm3", "_resid_ugm3")] = samples[col] / samples["EF"]
    used = long_segs.loc[np.unique(k[inside])]
    return samples, used
