"""Estimates of the ground (subsurface) heat flux G at NSA C1, where it is not measured.

The situation
-------------
ARM operates no soil heat-flux plates (SEBS), no soil temperature / moisture
profile (STAMP), and no snow temperature or snow depth sensor at the Barrow
Central Facility (every candidate datastream name was queried on 2026-09-15;
see seb.UNAVAILABLE_AT_C1). SEBS exists only at E10 Oliktok Point
("sebs_e10"). So at Barrow G has to be ESTIMATED, and every route below is
weaker than the radiative terms by an order of magnitude in reliability.

Sign convention: G is POSITIVE TOWARD THE SURFACE (heat arriving at the
snow-air interface from below), which is how it enters Sledd et al. (2025)
Eq. (1) with a plus sign. Note this is opposite to the usual heat-flux-plate
convention (positive downward into the soil).

Expected magnitude, for orientation (moderately confident, not a result):
over snow-covered tundra in Oct-Mar, G is a few W/m^2, occasionally reaching
of order 10-20 W/m^2 in October-November while the active layer refreezes and
the latent heat of freezing is conducted up through the snow (the "zero
curtain"); by February the profile is close to steady and G is small and
upward. Direct evidence from the nearest plates: the Oliktok Point SEBS
("sebs_e10") read G = +7.0 W/m^2 upward on 2025-12-15 with the 5-cm soil at
-3.9 degC and net radiation -7.8 W/m^2, closing its own budget to -0.8 W/m^2.
For comparison, the winter conductive flux through SHEBA sea ice was of order
10 W/m^2 (Persson et al. 2002). A G estimate that is large compared with
those numbers is more likely an error in another term than physics.

Three routes
------------
1. residual          G = -(NA)  where NA = LWD - LWU + SWD - SWU - SH - LH
                     (Eq. 5 with M = SWT = 0). Needs bulk SH and LH. Absorbs
                     the errors of EVERY other term, so it is a consistency
                     check on the budget rather than a measurement of G.
                     Averaging over hours-days reduces the random part.
2. snow conduction   G = k_snow (T_base - T_skin) / h_snow. Needs snow depth,
                     a temperature at the snow base (or in the soil) and a
                     snow thermal conductivity (from density via Sturm et al.
                     1997). None of these is an ARM measurement at C1; the
                     function exists so that a non-ARM source -- NOAA GML
                     Barrow Observatory, the NGEE-Arctic BEO site, CALM/GTN-P
                     permafrost boreholes -- can be plugged in. Whether any of
                     those covers 2025/26 has NOT been verified here.
3. thermal inertia   G from the skin-temperature history alone, treating the
                     subsurface as a homogeneous half-space (Wang & Bras
                     1999). Needs only T_skin (GNDIRT / LWU) and an assumed
                     thermal inertia sqrt(k rho c). Physically consistent and
                     independent of the other SEB terms, but the half-space
                     assumption is poor for a shallow snowpack over frozen
                     soil (two very different thermal inertias), and it
                     cannot represent the freezing-front latent heat release.

References
    Carslaw, H. S., & Jaeger, J. C. (1959). Conduction of Heat in Solids,
        2nd ed., Oxford UP (semi-infinite solid with prescribed surface
        temperature, Sect. 2.5).
    Persson, P. O. G., et al. (2002). JGR, 107(C10), 8045.
    Sturm, M., Holmgren, J., Konig, M., & Morris, K. (1997). The thermal
        conductivity of seasonal snow. J. Glaciol., 43(143), 26-41.
    Wang, J., & Bras, R. L. (1999). Ground heat flux estimated from surface
        soil temperature. J. Hydrology, 216, 214-226.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

# Specific heat of ice [J kg-1 K-1] near -20 degC.
C_ICE_J_KG_K = 2090.0


def snow_thermal_conductivity_sturm1997(rho_snow_kg_m3: np.ndarray) -> np.ndarray:
    """Effective thermal conductivity of seasonal snow [W m-1 K-1].

    Sturm et al. (1997) regression on 488 samples (their Eq. 4-5), with
    density rho in g cm-3:
        k = 0.138 - 1.01 rho + 3.233 rho^2      for 0.156 <= rho <= 0.6
        k = 0.023 + 0.234 rho                   for rho < 0.156
    Arctic tundra snow (wind slab over depth hoar) typically has bulk density
    0.25-0.35 g cm-3, giving k ~ 0.09-0.18 W m-1 K-1; depth hoar layers are
    much lower (~0.05). The scatter about the regression is roughly a factor
    of two, which propagates directly into G.
    """
    rho = np.asarray(rho_snow_kg_m3, dtype=float) / 1000.0
    return np.where(
        rho < 0.156, 0.023 + 0.234 * rho, 0.138 - 1.01 * rho + 3.233 * rho**2
    )


def ground_flux_residual(na_flux_w_m2: xr.DataArray) -> xr.DataArray:
    """G as the residual of Eq. (5) with M = SWT = 0: G = -NA.

    Parameters
    ----------
    na_flux_w_m2:
        seb.compute_seb_terms()["na_flux_W_m2"], i.e. LWD - LWU + SWD - SWU -
        SH - LH with bulk SH and LH supplied.

    Returns
    -------
    G [W m-2], positive toward the surface. Contains every other term's error;
    compare its magnitude with the expectations in the module docstring.
    """
    g = -na_flux_w_m2
    g.attrs = {
        "units": "W m-2",
        "long_name": "ground heat flux as Eq. (5) residual, positive toward surface",
        "caveat": "absorbs the errors of all radiative and bulk turbulent terms",
    }
    return g


def ground_flux_snow_conduction(
    t_skin_k: xr.DataArray,
    t_snow_base_k: xr.DataArray,
    snow_depth_m: xr.DataArray,
    k_snow_w_m_k: Optional[xr.DataArray] = None,
    rho_snow_kg_m3: Optional[float] = 300.0,
) -> xr.DataArray:
    """G by steady conduction through the snowpack: G = k (T_base - T_skin) / h.

    Parameters
    ----------
    t_skin_k, t_snow_base_k:
        Snow-surface and snow-base (soil-surface) temperatures [K].
    snow_depth_m:
        Snow depth [m]; the estimate diverges as h -> 0, so values < 2 cm
        are masked.
    k_snow_w_m_k:
        Snow conductivity; if None it is derived from `rho_snow_kg_m3` via
        Sturm et al. (1997).

    Returns
    -------
    G [W m-2], positive toward the surface (positive when the snow base is
    warmer than the surface, the normal winter state). Steady-state: ignores
    heat storage in the snow, which is small for a thin, cold pack but not
    negligible during rapid surface warming/cooling events (hours).
    """
    if k_snow_w_m_k is None:
        k = float(snow_thermal_conductivity_sturm1997(np.array(rho_snow_kg_m3)))
    else:
        k = k_snow_w_m_k
    h = snow_depth_m.where(snow_depth_m >= 0.02)
    g = k * (t_snow_base_k - t_skin_k) / h
    g.attrs = {
        "units": "W m-2",
        "long_name": "ground heat flux by steady snow conduction, positive toward surface",
        "k_snow_w_m_k": k if np.isscalar(k) else "array",
    }
    return g


def ground_flux_thermal_inertia(
    t_skin_k: xr.DataArray,
    thermal_inertia_j_m2_k_s05: float = 350.0,
    window: Optional[str] = None,
) -> xr.DataArray:
    """G from the skin-temperature history (Wang & Bras 1999, half-order derivative).

    For a homogeneous semi-infinite medium with a prescribed surface
    temperature T_s(t), the conductive flux INTO the medium at the surface is
    (Carslaw & Jaeger 1959, Sect. 2.5)

        q_down(t) = sqrt(k rho c / pi) * int_0^t [dT_s/ds] (t - s)^(-1/2) ds,

    i.e. the half-order time derivative of T_s scaled by the thermal inertia
    sqrt(k rho c). On a discrete series with steps Delta t the integral over
    each interval is analytic, giving (Wang & Bras 1999, their discretised
    form)

        q_down(t_n) = sqrt(k rho c / pi) * sum_{i=1}^{n}
                      2 (T_i - T_{i-1}) / [ sqrt(t_n - t_{i-1}) + sqrt(t_n - t_i) ].

    The Sledd-convention G is then -q_down (positive toward the surface).

    Parameters
    ----------
    t_skin_k:
        Skin temperature series [K] on a REGULAR time axis (resample first,
        e.g. hourly means; the sum is O(N^2), so 1-min data over a season is
        both slow and unnecessary). Gaps must be filled or the series split.
    thermal_inertia_j_m2_k_s05:
        sqrt(k rho c) [J m-2 K-1 s-1/2]. Snow at 300 kg m-3 with k = 0.2 and
        c = 2090 gives ~350; frozen mineral soil ~1500-2500. Because the
        near-surface medium is snow over soil, the effective value lies
        between and depends on how deep the thermal signal penetrates
        (diurnal ~ 10 cm in snow; the seasonal cooling ~ 1 m). This single
        number is the dominant uncertainty of the method.
    window:
        Optional pandas offset alias (e.g. "30D"): only temperature changes
        within this look-back window contribute. The kernel decays only as
        t^(-1/2), so truncation biases slowly varying components; None uses
        the full record.

    Returns
    -------
    G [W m-2], positive toward the surface, on the input time axis. The first
    value is NaN (no history). The initial transient (the medium is assumed
    isothermal at T_s(t_0)) decays over the first days of the record.
    """
    t = t_skin_k["time"].values
    if t.size < 3:
        raise ValueError("need at least 3 samples")
    dt_s = np.diff(t).astype("timedelta64[s]").astype(float)
    if not np.allclose(dt_s, dt_s[0], rtol=1e-3):
        raise ValueError(
            "time axis must be regular; resample t_skin first (e.g. hourly means)"
        )
    step = dt_s[0]
    ts = np.asarray(t_skin_k.values, dtype=float)
    if not np.isfinite(ts).all():
        raise ValueError("t_skin contains NaN; fill or split the series first")
    n = ts.size
    d_ts = np.diff(ts)  # T_i - T_{i-1}, i = 1..n-1
    q_down = np.full(n, np.nan)
    max_lag: Optional[int] = None
    if window is not None:
        # pandas offset alias -> seconds -> number of time steps to look back
        max_lag = int(window_to_seconds(window) / step)
    for k in range(1, n):
        i = np.arange(1, k + 1)  # intervals [t_{i-1}, t_i] with t_i <= t_k
        if max_lag is not None:
            i = i[(k - i + 1) <= max_lag]
        a = np.sqrt((k - i + 1) * step)  # sqrt(t_k - t_{i-1})
        b = np.sqrt((k - i) * step)  # sqrt(t_k - t_i)
        q_down[k] = np.sum(2.0 * d_ts[i - 1] / (a + b))
    q_down *= thermal_inertia_j_m2_k_s05 / np.sqrt(np.pi)
    g = xr.DataArray(-q_down, coords={"time": t_skin_k["time"]}, dims="time")
    g.attrs = {
        "units": "W m-2",
        "long_name": "ground heat flux from skin-temperature history (Wang & Bras 1999), positive toward surface",
        "thermal_inertia_j_m2_k_s05": thermal_inertia_j_m2_k_s05,
        "time_step_s": step,
        "caveat": "homogeneous half-space; no freezing latent heat; initial transient",
    }
    return g


def window_to_seconds(window: str) -> float:
    """Seconds in a pandas offset alias such as '30D'."""
    import pandas as pd

    return pd.Timedelta(window).total_seconds()
