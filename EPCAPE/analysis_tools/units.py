"""Unit conversions driven by each variable's ``units`` attribute.

The three cloud products report liquid water path in three different units
(MWRLOS ``liq`` in cm, MFRSRCLDOD ``lwp`` in mm, SPHOTCOD
``liquid_water_path`` in g m-2). Reading the unit from the file rather than
hard-coding it means a future reprocessing that changes units cannot
silently scale the data by 10 or 1000.

Water path of liquid water with density rho_w = 1000 kg m-3:
    1 mm of liquid = 1 kg m-2 = 1000 g m-2
    1 cm of liquid = 10 kg m-2 = 10 000 g m-2
"""

from __future__ import annotations

import re

import xarray as xr

# factor that multiplies a value in the given unit to give g m-2
_WATER_PATH_TO_GM2 = {
    "g/m2": 1.0,
    "g/m^2": 1.0,
    "gm-2": 1.0,
    "g m-2": 1.0,
    "g/m**2": 1.0,
    "g.m-2": 1.0,
    "kg/m2": 1e3,
    "kg/m^2": 1e3,
    "kgm-2": 1e3,
    "kg m-2": 1e3,
    "mm": 1e3,  # depth of liquid water (rho_w = 1000 kg m-3)
    "cm": 1e4,
    "um": 1.0,  # 1 micrometre of liquid = 1e-3 kg m-2 = 1 g m-2
}


_MICROMETRE = re.compile(r"^(microns?|micrometers?|micrometres?|μm|µm|um)$")


def _normalise(units: str) -> str:
    u = re.sub(r"\s+", " ", str(units).strip().lower())
    return "um" if _MICROMETRE.match(u) else u


def water_path_to_gm2(da: xr.DataArray) -> xr.DataArray:
    """A liquid (or vapour) water path converted to g m-2, units attribute updated.

    Raises ValueError for a unit it does not recognise, rather than guessing."""
    units = da.attrs.get("units")
    if units is None:
        raise ValueError(f"{da.name}: no units attribute; cannot convert water path safely.")
    key = _normalise(units)
    factor = _WATER_PATH_TO_GM2.get(key, _WATER_PATH_TO_GM2.get(key.replace(" ", "")))
    if factor is None:
        raise ValueError(f"{da.name}: unrecognised water-path unit {units!r}.")
    out = da * factor
    out.attrs = dict(da.attrs)
    out.attrs.update(units="g m-2", converted_from_units=str(units))
    return out


def radius_to_um(da: xr.DataArray) -> xr.DataArray:
    """A droplet radius converted to micrometres."""
    units = _normalise(da.attrs.get("units", "um"))
    factor = {"um": 1.0, "m": 1e6, "mm": 1e3, "cm": 1e4}.get(units)
    if factor is None:
        raise ValueError(f"{da.name}: unrecognised radius unit {da.attrs.get('units')!r}.")
    out = da * factor
    out.attrs = dict(da.attrs)
    out.attrs.update(units="um", converted_from_units=str(da.attrs.get("units", "um (assumed)")))
    return out
