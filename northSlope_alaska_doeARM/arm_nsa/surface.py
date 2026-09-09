"""Surface meteorology (MET/METTWR) and broadband radiation (QCRAD) readers.

These datastreams are not in the Hartig26 Table 1 instrument set, but they are
the backbone of Bertrand25's radiation-environment analysis (config.py role
"bertrand25-core") and of the Ocean Visions MPCT surface energy budget
questions: "would freezing supercooled liquid cause substantial cooling at the
surface?" Answering that against observations requires the downwelling
longwave flux and the 2-m air temperature, which at NSA come from:

* MET  (nsametC1.b1): 1-min conventional surface sensors -- 2-m temperature
  and humidity, winds, station pressure.
* QCRAD1LONG (nsaqcrad1longC1.c2/.c1): the ARM "best estimate" quality-
  controlled broadband fluxes -- down/up shortwave and longwave [W/m^2].
  During polar night the shortwave channels are ~0 by physics, and the
  cloud signal lives almost entirely in the downwelling longwave: the
  canonical bimodal Arctic winter surface state (radiatively clear vs.
  opaquely cloudy, Stramler et al. 2011) is separated by roughly
  30-60 W/m^2 in lwdn_w_m2.

Both readers return tidy time-indexed Datasets in pipeline-canonical units
(temperatures degC, fluxes W/m^2, pressure kPa).
"""

from __future__ import annotations

import xarray as xr

from .readers import read_timeseries


def read_met(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read 1-min surface meteorology for a date range.

    Returns
    -------
    Dataset over time with:
        temp_2m_c   2-m air temperature [degC]
        rh_2m_pct   2-m relative humidity [%]
        wspd_m_s    wind speed [m/s]
        wdir_deg    wind direction [deg from N]
        pres_kpa    station pressure [kPa]
    """
    ds = read_timeseries(
        "met", start_date, end_date, apply_qc_flags=apply_qc_flags, verbose=verbose
    )
    ds["temp_2m_c"].attrs.setdefault("units", "degC")
    ds["temp_2m_c"].attrs["long_name"] = "2-m air temperature"
    return ds


def read_mettwr(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read METTWR (tower met, ~1998-2003) 2-m temperature and humidity.

    METTWR predates MET at NSA; Bertrand25 splices METTWR + MET to get a
    26-year surface record. Use read_met() from late 2003 onward. The 2-m
    variable names in this older ingest are asserted via candidate lists in
    config.py -- if the reader raises a KeyError listing the file's variables,
    add the correct name there (one-line fix).
    """
    ds = read_timeseries(
        "mettwr", start_date, end_date, apply_qc_flags=apply_qc_flags, verbose=verbose
    )
    ds["temp_2m_c"].attrs.setdefault("units", "degC")
    ds["temp_2m_c"].attrs["long_name"] = "2-m air temperature (tower met)"
    return ds


def read_qcrad(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read best-estimate broadband surface radiative fluxes for a date range.

    Returns
    -------
    Dataset over time with (sign convention: all fluxes positive):
        swdn_w_m2   downwelling shortwave, global hemispheric [W/m^2]
        lwdn_w_m2   downwelling longwave [W/m^2]
        swup_w_m2   upwelling shortwave [W/m^2]
        lwup_w_m2   upwelling longwave [W/m^2]

    The surface net radiation is then
        net = (swdn - swup) + (lwdn - lwup)
    which in polar night reduces to the longwave terms.
    """
    ds = read_timeseries(
        "qcrad", start_date, end_date, apply_qc_flags=apply_qc_flags, verbose=verbose
    )
    for name, long_name in [
        ("swdn_w_m2", "downwelling shortwave irradiance (best estimate)"),
        ("lwdn_w_m2", "downwelling longwave irradiance"),
        ("swup_w_m2", "upwelling shortwave irradiance"),
        ("lwup_w_m2", "upwelling longwave irradiance"),
    ]:
        ds[name].attrs.setdefault("units", "W/m^2")
        ds[name].attrs["long_name"] = long_name
    return ds


def read_tower_winds(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read 1-min 40-m tower meteorology (nsatwrC1.b1) for a date range.

    This is the wind source for the Taylor frozen-turbulence cloud-scale
    estimate. Unlike every other reader in this module the fields are
    two-dimensional -- (time, height) -- because the tower reports all four of
    its levels in one variable rather than under per-level names.

    Returns
    -------
    Dataset over (time, height) with:
        wspd_arith_m_s  1-min arithmetic-mean wind speed [m/s]
        wspd_vec_m_s    1-min vector-mean wind speed [m/s]
        wdir_deg        1-min vector-mean wind direction [deg from N]
        temp_c          air temperature [degC]
        rh_pct          relative humidity [%]
    with `height` = 2, 10, 20, 40 m AGL (config.TOWER_HEIGHTS_M).

    Notes
    -----
    Which wind speed to use depends on the question. `wspd_arith_m_s` is the
    mean SPEED over the minute; `wspd_vec_m_s` is the magnitude of the mean
    VELOCITY, i.e. net displacement per unit time, and is therefore the
    quantity Taylor's hypothesis wants when converting a duration into a
    distance. The two coincide unless the wind direction swings within the
    averaging interval, which at NSA in winter is uncommon.

    The 10-m level is the same measurement nsametC1.b1 reports as its single
    wind level (same ingest, same `input_source`), so read_met() is a
    cross-check on this reader, not an independent sample.
    """
    ds = read_timeseries(
        "twr", start_date, end_date, apply_qc_flags=apply_qc_flags, verbose=verbose
    )
    for name, long_name in [
        ("wspd_arith_m_s", "wind speed, 1-min arithmetic mean"),
        ("wspd_vec_m_s", "wind speed, 1-min vector mean"),
    ]:
        ds[name].attrs.setdefault("units", "m/s")
        ds[name].attrs["long_name"] = long_name
    ds["wdir_deg"].attrs.setdefault("units", "degree")
    ds["temp_c"].attrs.setdefault("units", "degC")
    ds["height"].attrs.setdefault("units", "m")
    ds["height"].attrs["long_name"] = "measurement height above ground level"
    return ds


def tower_level(ds: xr.Dataset, height_m: float) -> xr.Dataset:
    """Select one tower level by its height in metres, e.g. tower_level(ds, 40).

    Raises KeyError listing the available heights rather than returning the
    nearest one -- silently sliding from 40 m to 20 m would change the answer
    without changing the plot label.
    """
    available = [float(h) for h in ds["height"].values]
    if float(height_m) not in available:
        raise KeyError(
            f"No tower level at {height_m} m. Available heights [m]: {available}"
        )
    return ds.sel(height=height_m)
