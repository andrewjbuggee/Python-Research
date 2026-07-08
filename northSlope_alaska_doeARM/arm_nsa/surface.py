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
