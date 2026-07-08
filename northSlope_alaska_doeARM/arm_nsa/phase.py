"""Cloud phase / sky-state classification for sonde-coordinated samples.

Ground-based instruments cannot measure phase directly; what exists are
consistent proxies (Hartig26; Shupe 2011; Silber et al. 2020):

* MWR LWP >= 10 g/m^2         -> liquid is certainly present (97% of such
                                  cases have a saturated layer overhead).
* sonde saturated layer        -> proxy for a liquid-containing region;
                                  at LWP 0-10 g/m^2, 65% of cases still have
                                  a saturated layer (thin liquid clouds).
* KAZR reflectivity            -> hydrometeors of any phase (cloud + precip).
* KAZR clear-sky flag          -> no significant hydrometeors below 10 km.

This module combines them into one categorical variable per sounding. The
categories are deliberately conservative about what can actually be known:

    0 MISSING          not enough data to classify (no usable radar window
                        AND no MWR retrieval)
    1 CLEAR            radar clear-sky flag set, and LWP below the clear-sky
                        threshold or missing
    2 ICE_PROBABLE     hydrometeors present, LWP < 10 g/m^2, and NO saturated
                        layer in the sounding -> likely all-ice cloud/precip
    3 LIQUID_PROBABLE  hydrometeors present, LWP < 10 g/m^2, but a saturated
                        layer exists -> likely thin/supercooled liquid layer
                        (the MPCT-relevant marginal cases)
    4 LIQUID_CONFIDENT LWP >= 10 g/m^2 -> unambiguously liquid-containing
                        (mixed-phase in winter almost surely: ice usually
                        coexists at these temperatures)

Precedence: LIQUID_CONFIDENT > CLEAR > (ICE|LIQUID)_PROBABLE > MISSING.

For MPCT purposes, categories 3+4 together estimate the frequency of
"seedable" liquid-containing scenes; category 4 alone is the lower bound.

NOTE: where the Shupe-Turner microphysics product is available (~2004-2019),
its lidar-depolarization-informed, vertically resolved phase classification
is strictly stronger than this heuristic -- prefer arm_nsa.shupe_turner
(clear / ice_only / mixed_phase / liquid_only scenes, Bertrand25 rules) there,
and use this module to extend phase statistics outside that window or as a
consistency check.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import xarray as xr

from . import config

PHASE_CODES: Dict[int, str] = {
    0: "missing",
    1: "clear",
    2: "ice_probable",
    3: "liquid_probable",
    4: "liquid_confident",
}


def classify_phase(library: xr.Dataset) -> xr.DataArray:
    """Classify each sonde-coordinated sample into a sky-state category.

    Parameters
    ----------
    library:
        Output of coordinate.build_library(); needs lwp_g_m2, clear_sky,
        radar_coverage_fraction, n_saturated_layers, ceil_cloud_fraction.

    Returns
    -------
    Integer DataArray over launch_time with the PHASE_CODES mapping in attrs.
    """
    lwp_g_m2 = library["lwp_g_m2"].values
    clear_flag = library["clear_sky"].values.astype(bool)
    radar_cov = library["radar_coverage_fraction"].values
    n_sat_layers = library["n_saturated_layers"].values
    ceil_frac = library["ceil_cloud_fraction"].values

    lwp_valid = np.isfinite(lwp_g_m2)
    lwp_thresh = config.LWP_CLEAR_SKY_THRESHOLD_G_M2
    radar_usable = np.isfinite(radar_cov) & (
        radar_cov >= config.RADAR_MIN_COVERAGE_FRACTION
    )
    # Hydrometeor presence: prefer the radar; fall back to the ceilometer
    # when the radar window was unusable (ceilometer sees the low clouds
    # that matter most here, though it saturates in obscuring snow).
    hydrometeors = np.where(
        radar_usable,
        ~clear_flag,
        np.isfinite(ceil_frac) & (ceil_frac > 0.05),
    )

    codes = np.zeros(lwp_g_m2.shape, dtype=np.int8)  # default 0 = missing

    classifiable = radar_usable | lwp_valid | np.isfinite(ceil_frac)
    # CLEAR: radar says clear and the MWR does not contradict it.
    is_clear = radar_usable & clear_flag & (~lwp_valid | (lwp_g_m2 < lwp_thresh))
    # LIQUID_CONFIDENT: MWR alone is sufficient.
    is_liq_conf = lwp_valid & (lwp_g_m2 >= lwp_thresh)
    # Remaining cloudy cases split on the sounding's saturated layers.
    cloudy_low_lwp = classifiable & hydrometeors & ~is_liq_conf
    is_liq_prob = cloudy_low_lwp & (n_sat_layers > 0)
    is_ice_prob = cloudy_low_lwp & (n_sat_layers == 0)

    codes[is_ice_prob] = 2
    codes[is_liq_prob] = 3
    codes[is_clear] = 1  # after probable, before confident: precedence
    codes[is_liq_conf] = 4

    out = xr.DataArray(
        codes,
        dims=("launch_time",),
        coords={"launch_time": library["launch_time"]},
        name="phase_code",
    )
    out.attrs.update(
        long_name="sky-state / cloud phase classification",
        flag_values=list(PHASE_CODES.keys()),
        flag_meanings=" ".join(PHASE_CODES.values()),
        lwp_threshold_g_m2=config.LWP_CLEAR_SKY_THRESHOLD_G_M2,
    )
    return out


def phase_occurrence(phase_code: xr.DataArray) -> Dict[str, float]:
    """Occurrence fraction per category, over classified (non-missing) samples.

    Returns a dict like {"clear": 0.28, "ice_probable": 0.12, ...} plus
    "liquid_containing_any" = liquid_probable + liquid_confident, the quantity
    comparable to Hartig26's 60-70% November-March liquid occurrence.
    """
    codes = phase_code.values
    classified = codes != 0
    n = int(classified.sum())
    out: Dict[str, float] = {"n_classified": float(n)}
    if n == 0:
        return out
    for code, name in PHASE_CODES.items():
        if code == 0:
            continue
        out[name] = float((codes[classified] == code).mean())
    out["liquid_containing_any"] = out.get("liquid_probable", 0.0) + out.get(
        "liquid_confident", 0.0
    )
    return out
