"""Assemble the 1-min analysis table for the DLR / cloud-microphysics notebook.

Joins, on a canonical 1-min grid:

  * THERMOCLDPHASE column features (arm_nsa.column_features; 30 s -> the
    sample nearest each minute, tolerance 20 s, i.e. the :00 profile)
  * QCRAD broadband fluxes (1 min; Bad-QC removed): lwdn/lwup/swdn/swup
  * MET 2-m temperature, RH, wind, pressure (1 min)
  * GNDIRT surface IR brightness temperature (1 min) and the skin temperature
    derived from it and from LWU
  * CLDTYPE cloud-type classification of the lowest layer and precipitation
    rate (1 min; optional)
  * MICROBASE 1-min column reductions from scripts/reduce_microbase.py on the
    sampled days (optional; NaN elsewhere), variables prefixed "mb_"

Skin temperature conventions (both kept so their difference can be used as a
screen, see README "Skin temperature: IRT vs LWU disagree"):
    t_skin_irt_K   IRT brightness temperature corrected for snow emissivity
                   0.985 and reflected sky (seb.skin_temperature_from_irt)
    t_skin_lwu_K   inverted from LWU: LWU = eps sigma T^4 + (1 - eps) LWD
                   with eps = 0.985 (Sledd et al. 2025, Sect. 2.1)

The result is written to data/processed/dlr_microphysics_1min_<start>_<end>.nc
and re-used by the notebook; rebuild with force=True after new raw days land.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .seb import read_cldtype, read_gndirt, skin_temperature_from_irt
from .surface import read_met, read_qcrad

SIGMA_SB_W_M2_K4 = 5.670374419e-8  # Stefan-Boltzmann constant [W m-2 K-4], CODATA 2018
SNOW_EMISSIVITY = 0.985  # Sledd et al. (2025) and this package's SEB module


def canonical_minutes(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Every minute from start 00:00 to end 23:59 inclusive."""
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return pd.date_range(pd.Timestamp(start_date), end, freq="1min", name="time")


def _align(ds: xr.Dataset, minutes: pd.DatetimeIndex, tolerance_s: float) -> xr.Dataset:
    """Nearest-neighbour reindex onto the minute grid within a tolerance."""
    if ds.sizes.get("time", 0) == 0:
        return ds
    ds = ds.sortby("time")
    _, idx = np.unique(ds["time"].values, return_index=True)
    if idx.size != ds.sizes["time"]:
        ds = ds.isel(time=idx)
    return ds.reindex(time=minutes, method="nearest", tolerance=np.timedelta64(int(tolerance_s * 1e9), "ns"))


def _columns_dir(lidar: str) -> Path:
    return config.PROCESSED_DATA_DIR / "dlr_columns" / lidar


def load_columns(start_date: str, end_date: str, lidar: str = "mplgr") -> xr.Dataset:
    """Concatenate the per-day column-feature files in a date range."""
    d = _columns_dir(lidar)
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    paths = sorted(p for p in d.glob("dlr_columns.*.nc") if s <= p.name.split(".")[1] <= e)
    if not paths:
        raise FileNotFoundError(
            f"No column files in {d} for {start_date}..{end_date}; run "
            f"scripts/build_dlr_columns.py --start {start_date} --end {end_date}"
        )
    pieces = [xr.open_dataset(p) for p in paths]
    out = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
    for p in pieces:
        p.close()
    out.attrs["n_column_days"] = len(paths)
    return out


def load_microbase_1min(start_date: str, end_date: str) -> Optional[xr.Dataset]:
    """Column part of the reduced MICROBASE days in range (None if none)."""
    d = config.PROCESSED_DATA_DIR / "microbase_1min"
    s = start_date.replace("-", "")
    e = end_date.replace("-", "")
    paths = sorted(p for p in d.glob("nsamicrobase_1min.*.nc") if s <= p.name.split(".")[1] <= e)
    if not paths:
        return None
    pieces = []
    for p in paths:
        ds = xr.open_dataset(p)
        pieces.append(ds[[v for v in ds.data_vars if ds[v].ndim == 1]].load())
        ds.close()
    out = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
    out = out.rename({v: f"mb_{v}" for v in out.data_vars})
    out.attrs["n_microbase_days"] = len(paths)
    return out


def build_dlr_dataset(
    start_date: str,
    end_date: str,
    lidar: str = "mplgr",
    with_cldtype: bool = True,
    with_microbase: bool = True,
    out_path: Optional[Path] = None,
    force: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """Build (or load) the merged 1-min table for one date range."""
    out_path = Path(out_path) if out_path else (
        config.PROCESSED_DATA_DIR / f"dlr_microphysics_1min_{start_date}_{end_date}_{lidar}.nc"
    )
    if out_path.exists() and not force:
        if verbose:
            print(f"loading {out_path.name}")
        return xr.open_dataset(out_path).load()

    minutes = canonical_minutes(start_date, end_date)
    parts = []

    cols = load_columns(start_date, end_date, lidar)
    n_col_days = cols.attrs.get("n_column_days", 0)
    parts.append(_align(cols, minutes, 20.0))
    if verbose:
        print(f"columns: {n_col_days} days")

    qcrad = read_qcrad(start_date, end_date)
    parts.append(_align(qcrad, minutes, 30.0))
    if verbose:
        print(f"qcrad: {qcrad.sizes['time']} samples")

    try:
        met = read_met(start_date, end_date)
        parts.append(_align(met, minutes, 30.0))
    except FileNotFoundError as err:
        print(f"met missing: {err}")
    try:
        irt = read_gndirt(start_date, end_date)
        parts.append(_align(irt, minutes, 30.0))
    except FileNotFoundError as err:
        print(f"gndirt missing: {err}")

    if with_cldtype:
        try:
            ct = read_cldtype(start_date, end_date, include_reflectivity=False)
            keep = xr.Dataset()
            ctype = ct["cloud_type"]
            keep["cloudtype_layer1"] = ctype.isel(layer=0)
            keep["cloudtype_max"] = ctype.max("layer", skipna=True)
            keep["precip_rate_mm_min"] = ct["precip_rate_mm_min"]
            keep["cloud_base_best_estimate_m"] = ct["cloud_base_best_estimate_m"]
            for v in keep.data_vars:
                keep[v].attrs = dict(ct[v.replace("cloudtype_layer1", "cloud_type").replace("cloudtype_max", "cloud_type")].attrs)
            parts.append(_align(keep, minutes, 30.0))
        except FileNotFoundError as err:
            print(f"cldtype missing: {err}")

    if with_microbase:
        mb = load_microbase_1min(start_date, end_date)
        if mb is not None:
            parts.append(_align(mb, minutes, 30.0))
            if verbose:
                print(f"microbase: {mb.attrs.get('n_microbase_days')} sampled days")

    ds = xr.merge(parts, compat="override", combine_attrs="drop_conflicts")

    # ---- derived surface temperatures ------------------------------------------
    if "temp_2m_c" in ds:
        ds["t_air_2m_K"] = ds["temp_2m_c"] + 273.15
        ds["t_air_2m_K"].attrs = {"units": "K", "long_name": "2-m air temperature (MET)"}
    lwdn = ds["lwdn_w_m2"]
    lwup = ds["lwup_w_m2"]
    t_lwu = ((lwup - (1.0 - SNOW_EMISSIVITY) * lwdn) / (SNOW_EMISSIVITY * SIGMA_SB_W_M2_K4)) ** 0.25
    ds["t_skin_lwu_K"] = t_lwu
    ds["t_skin_lwu_K"].attrs = {
        "units": "K",
        "long_name": f"skin temperature inverted from LWU with emissivity {SNOW_EMISSIVITY}",
    }
    if "t_skin_ir_k" in ds:
        ds["t_skin_irt_K"] = skin_temperature_from_irt(ds["t_skin_ir_k"], lwdn, SNOW_EMISSIVITY)

    ds.attrs = {
        "title": "DLR / cloud-microphysics 1-min analysis table (arm_nsa.dlr_dataset)",
        "site": "ARM NSA C1, Utqiagvik AK",
        "start_date": start_date,
        "end_date": end_date,
        "lidar": lidar,
        "n_column_days": int(n_col_days),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 3} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=enc)
    if verbose:
        print(f"wrote {out_path.name}: {ds.sizes['time']} minutes, {len(ds.data_vars)} variables")
    return ds
