"""Cloud water paths and cloud temperatures from the NSA SEB datastreams.

These are the observational analogues of the ERA5 cloud-state fields the SEB
comparison uses (tclw, tciw, and the pressure-level cloud temperature):

    LWP   MWRRET be_lwp (mwr.read_mwr) and/or MWR3C (seb.read_mwr3c). Nothing
          to compute -- both are column retrievals. merge_lwp() below just
          picks the best available one per time step.
    IWP   No routine column retrieval at NSA short of MICROBASE (670 MB/day).
          iwp_from_reflectivity() integrates the reflectivity-based IWC
          relation already used by radar.py over the 1-min ARSCL profile that
          CLDTYPE carries (9 MB/day), restricted to ice-containing gates when
          a THERMOCLDPHASE mask is supplied.
    T_cld cloud_temperature_at_boundaries() places the ARSCL layer base/top
          heights in a temperature profile (THERMOCLDPHASE 'sonde_temp', the
          INTERPSONDE 'temp' field, or a sonde library) and reports the
          temperature at each boundary.

IWC-Z relation and its uncertainty
----------------------------------
    IWC [g m-3] = a Z^b,  Z in mm^6 m-3 (linear, NOT dBZ)
with a = 0.1 (SHEBA winter average, Shupe et al. 2005) and b = 0.63
(Matrosov 1999), i.e. config.IWC_PREFACTOR_A / IWC_EXPONENT_B. Published
Arctic coefficients vary by more than a factor of two in `a`; Shupe et al.
(2005, J. Appl. Meteor. 44, 1544-1562) quote an IWC uncertainty of roughly a
factor of 2, and MICROBASE, which uses the same family of relations, is not
independent of this estimate. Treat the IWP as an order-of-magnitude
quantity, adequate for separating ice-dominated from liquid-dominated scenes
and for a comparison with ERA5 tciw, not for closing a water budget.

Liquid gates contribute little: a non-precipitating liquid layer has
Z < -20 dBZ, i.e. Z < 0.01 mm^6 m-3 and IWC < 0.006 g m-3, so even without a
phase mask a 300-m liquid layer adds < 2 g m-2. Drizzle and snow are the
cases where the mask matters.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import xarray as xr

from . import config


def iwc_from_reflectivity(
    reflectivity_dbz: xr.DataArray,
    a: float = config.IWC_PREFACTOR_A,
    b: float = config.IWC_EXPONENT_B,
) -> xr.DataArray:
    """IWC [g m-3] = a Z^b with Z converted from dBZ to mm^6 m-3."""
    z_lin = 10.0 ** (reflectivity_dbz / 10.0)
    iwc = a * z_lin**b
    iwc.attrs = {
        "units": "g m-3",
        "long_name": f"ice water content from reflectivity, IWC = {a} Z^{b}",
    }
    return iwc


def iwp_from_reflectivity(
    reflectivity_dbz: xr.DataArray,
    phase_mask: Optional[xr.DataArray] = None,
    ice_codes: Sequence[int] = config.THERMO_ICE_CONTAINING_CODES,
    min_dbz: float = -40.0,
    max_height_m: float = 12_000.0,
    a: float = config.IWC_PREFACTOR_A,
    b: float = config.IWC_EXPONENT_B,
) -> xr.DataArray:
    """Ice water path [g m-2] by vertical integration of IWC(Z).

    Parameters
    ----------
    reflectivity_dbz:
        (time, height) reflectivity with a `height` coordinate in metres
        (CLDTYPE 'reflectivity_dbz' via seb.read_cldtype, or ARSCL/KAZR).
        NaN where nothing was detected.
    phase_mask:
        Optional (time, height) THERMOCLDPHASE pixel phase codes on the SAME
        height grid (both products use 596 gates at 30 m, 160-18010 m AGL).
        Only gates whose code is in `ice_codes` (default: ice, mixed, snow --
        config.THERMO_ICE_CONTAINING_CODES) contribute. Aligned to the
        reflectivity time axis by nearest neighbour within 60 s.
    min_dbz:
        Gates below this reflectivity are ignored (noise floor).
    max_height_m:
        Integration ceiling; the 18-km grid top is well above any ice cloud
        that matters for the surface budget.

    Returns
    -------
    IWP [g m-2] on the reflectivity time axis; 0 where no ice gate exists and
    NaN where the whole profile is missing.
    """
    z = reflectivity_dbz.where(reflectivity_dbz >= min_dbz)
    z = z.sel(height=slice(None, max_height_m))
    if phase_mask is not None:
        pm = phase_mask.sel(height=slice(None, max_height_m))
        pm = pm.reindex(
            time=z["time"], method="nearest", tolerance=np.timedelta64(60, "s")
        )
        if not np.allclose(pm["height"].values, z["height"].values):
            pm = pm.interp(height=z["height"], method="nearest")
        is_ice = xr.zeros_like(pm, dtype=bool)
        for code in ice_codes:
            is_ice = is_ice | (pm == code)
        z = z.where(is_ice)
    iwc = iwc_from_reflectivity(z, a, b)
    # Gate thickness from the (uniform) height grid.
    h = z["height"].values
    dz = np.gradient(h) if h.size > 1 else np.array([config.RADAR_GRID_STEP_M])
    iwp = (iwc.fillna(0.0) * xr.DataArray(dz, coords={"height": h}, dims="height")).sum(
        "height"
    )
    all_missing = ~np.isfinite(
        reflectivity_dbz.sel(height=slice(None, max_height_m))
    ).any("height")
    iwp = iwp.where(~all_missing)
    iwp.attrs = {
        "units": "g m-2",
        "long_name": "ice water path from reflectivity (IWC = a Z^b, integrated)",
        "a": a,
        "b": b,
        "min_dbz": min_dbz,
        "phase_masked": phase_mask is not None,
        "uncertainty": "factor ~2 (Z-IWC relation); see cloud_water.py",
    }
    return iwp


def cloud_temperature_at_boundaries(
    layer_base_m: xr.DataArray,
    layer_top_m: xr.DataArray,
    temp_profile: xr.DataArray,
    temp_units: str = "auto",
) -> xr.Dataset:
    """Temperature at each ARSCL layer base and top from a (time, height) profile.

    Parameters
    ----------
    layer_base_m, layer_top_m:
        (time, layer) heights [m AGL] with NaN for absent layers
        (seb.read_arscl_boundaries or seb.read_cldtype output).
    temp_profile:
        (time, height) temperature with `height` in METRES, e.g.
        THERMOCLDPHASE 'sonde_temp' (note its height coordinate is in km --
        convert first: config.THERMO_HEIGHT_UNITS), or INTERPSONDE 'temp'.
        Aligned to the boundary time axis by nearest neighbour within 30 min
        (the profile is a 1-min interpolation of ~6-hourly launches, so this
        loses nothing real).
    temp_units:
        "degC", "K", or "auto" (uses the `units` attribute; degC assumed if
        absent). Output is in K.

    Returns
    -------
    Dataset over (time, layer) with t_base_K, t_top_K, and over time with
    t_lowest_base_K / t_lowest_top_K (the first layer), which is the cloud
    that controls the surface longwave.
    """
    units = (
        temp_profile.attrs.get("units", "degC") if temp_units == "auto" else temp_units
    )
    offset = 273.15 if str(units).lower().startswith(("degc", "c")) else 0.0
    prof = temp_profile.reindex(
        time=layer_base_m["time"], method="nearest", tolerance=np.timedelta64(30, "m")
    )
    # interp() attaches the (time, layer) target heights as a `height`
    # coordinate on each result; drop them so base and top can share a Dataset.
    t_base = (prof.interp(height=layer_base_m) + offset).drop_vars(
        "height", errors="ignore"
    )
    t_top = (prof.interp(height=layer_top_m) + offset).drop_vars(
        "height", errors="ignore"
    )
    out = xr.Dataset(
        {
            "t_base_K": t_base,
            "t_top_K": t_top,
            "t_lowest_base_K": t_base.isel(layer=0),
            "t_lowest_top_K": t_top.isel(layer=0),
        }
    )
    for name in out.data_vars:
        out[name].attrs = {"units": "K", "long_name": name.replace("_", " ")}
    return out


def merge_lwp(
    primary_g_m2: xr.DataArray,
    secondary_g_m2: xr.DataArray,
    time: Optional[xr.DataArray] = None,
    tolerance_s: float = 90.0,
) -> xr.Dataset:
    """Best-available LWP per time step: primary (MWRRET) where finite, else secondary (MWR3C).

    Returns lwp_g_m2 plus lwp_source (1 primary, 2 secondary, 0 none) so the
    provenance survives averaging. Both inputs are nearest-neighbour aligned
    to `time` (default: the primary's axis).
    """
    t = primary_g_m2["time"] if time is None else time
    tol = np.timedelta64(int(tolerance_s), "s")
    p = primary_g_m2.reindex(time=t, method="nearest", tolerance=tol)
    s = secondary_g_m2.reindex(time=t, method="nearest", tolerance=tol)
    use_p = np.isfinite(p)
    lwp = p.where(use_p, s)
    source = xr.where(use_p, 1, xr.where(np.isfinite(s), 2, 0)).astype("int8")
    out = xr.Dataset({"lwp_g_m2": lwp, "lwp_source": source})
    out["lwp_g_m2"].attrs = {
        "units": "g/m^2",
        "long_name": "liquid water path, best available",
    }
    out["lwp_source"].attrs = {
        "flag_values": "0 1 2",
        "flag_meanings": "none primary secondary",
    }
    return out
