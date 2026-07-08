"""Microwave radiometer retrievals (MWRRET) of LWP and PWV.

The MWRRET "liljclou" value-added product (Turner et al. 2007) provides
best-estimate liquid water path and precipitable water vapor from a physical
retrieval on the 23.8 / 31.4 GHz brightness temperatures, including clear-sky
brightness-temperature offset corrections. Hartig26 Table 1 uses
nsamwrret1liljclouC1.c2 at 20-30 s resolution.

Retrieval uncertainty context (Hartig26 Sect. 2): the theoretical LWP
uncertainty is ~25 g/m^2, but clear-sky retrievals overwhelmingly sit within a
few g/m^2 of zero, so in practice LWP below ~10 g/m^2 is treated as
indistinguishable from clear sky (config.LWP_CLEAR_SKY_THRESHOLD_G_M2) and
analyses either use full distributions or split out the 0-10 g/m^2 bin.

Canonical units here: LWP in g/m^2, PWV in cm. Conversions are applied from
the file's units attribute when needed (older MWR products report liquid in
cm: 1 cm of liquid water = 10^4 g/m^2; 1 mm = 10^3 g/m^2).
"""

from __future__ import annotations

import xarray as xr

from .readers import read_timeseries

# units-attribute string (lowercased, stripped) -> multiplicative factor
_LWP_TO_G_M2 = {
    "g/m^2": 1.0,
    "g/m2": 1.0,
    "g m^-2": 1.0,
    "kg/m^2": 1.0e3,
    "kg/m2": 1.0e3,
    "mm": 1.0e3,
    "cm": 1.0e4,
}
_PWV_TO_CM = {
    "cm": 1.0,
    "mm": 0.1,
    "kg/m^2": 0.1,
    "kg/m2": 0.1,
}


def _convert_units(arr: xr.DataArray, table: dict, target_units: str) -> xr.DataArray:
    units = str(arr.attrs.get("units", "")).strip().lower()
    factor = table.get(units)
    if factor is None:
        # Unknown/absent units string: leave values untouched but say so, so a
        # unit surprise is visible in the metadata instead of silent.
        arr.attrs["units_note"] = (
            f"file units {units!r} not recognized; values passed through unconverted"
        )
        return arr
    out = arr * factor
    out.attrs = dict(arr.attrs)
    out.attrs["units"] = target_units
    if factor != 1.0:
        out.attrs["units_note"] = f"converted from {units!r} (factor {factor:g})"
    return out


def read_mwr(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read MWRRET best-estimate LWP and PWV for a date range.

    Returns
    -------
    Dataset over time with:
        lwp_g_m2  liquid water path [g/m^2]
        pwv_cm    precipitable water vapor [cm]
    QC-flagged retrievals arrive as NaN (apply_qc_flags=True). Note that ~22%
    of sonde-coordinated hours in Hartig26 lack MWR data entirely; gaps are
    normal for this instrument.
    """
    ds = read_timeseries(
        "mwr",
        start_date,
        end_date,
        apply_qc_flags=apply_qc_flags,
        verbose=verbose,
    )
    ds["lwp_g_m2"] = _convert_units(ds["lwp_g_m2"], _LWP_TO_G_M2, "g/m^2")
    ds["lwp_g_m2"].attrs.setdefault("long_name", "liquid water path (best estimate)")
    ds["pwv_cm"] = _convert_units(ds["pwv_cm"], _PWV_TO_CM, "cm")
    ds["pwv_cm"].attrs.setdefault(
        "long_name", "precipitable water vapor (best estimate)"
    )
    return ds
