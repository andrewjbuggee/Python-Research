"""Sonde-coordinated multi-instrument library (Hartig26 Sect. 2).

The different NSA instruments run on wildly different clocks (sonde: 0-4
launches/day; KAZR: ~5 s; MWR: ~25 s; ceilometer: 15 s). Hartig26 puts them on
a common footing by using the radiosonde launches as anchor points: for each
launch, every other instrument is averaged over the HOUR FOLLOWING the launch
time. Missing data are ignored inside the average, except that a radar hourly
mean requires at least half of its time samples to carry reflectivity.

This module builds that per-launch "library" as a single xarray Dataset:

    dims:   launch_time, height (sonde 5-m grid), radar_height (30-m grid)
    sonde:  tdry_c, rh_pct, ... profiles + saturated-layer summary
    mwr:    lwp_g_m2, pwv_cm window means + sample counts
    ceil:   first_cbh_m window mean + detection fraction
    kazr:   cloud_frac_profile, iwp_g_m2, clear_sky, radar coverage
    [option] met/qcrad window means (temp_2m_c, lwdn_w_m2, ...) -- extension
             beyond Hartig26, for surface energy budget context.

KAZR data are read one launch-window at a time (they are far too large to
hold a season in memory at native resolution); everything else is read once
for the full period.

The output of build_library() is what the statistics / plotting layers should
consume -- one row per sounding, all instruments aligned.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .ceilometer import read_ceilometer
from .mwr import read_mwr
from .radar import (
    clear_sky_flag,
    cloud_fraction_profile,
    iwp_g_m2,
    radar_height_grid_m,
    read_kazr,
    window_coverage_fraction,
)
from .sonde import read_sondes, saturated_layer_summary
from .surface import read_met, read_qcrad


def _window_mean(
    ds: Optional[xr.Dataset],
    var: str,
    t0: np.datetime64,
    t1: np.datetime64,
) -> tuple:
    """(mean, n_valid) of `var` in [t0, t1); NaN/0 when data are absent."""
    if ds is None or var not in ds:
        return float("nan"), 0
    window = ds[var].sel(time=slice(t0, t1))
    n_valid = int(np.isfinite(window).sum())
    if n_valid == 0:
        return float("nan"), 0
    return float(window.mean()), n_valid


def build_library(
    start_date: str,
    end_date: str,
    include_extensions: bool = True,
    skip_radar: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """Build the sonde-coordinated multi-instrument dataset for a date range.

    Parameters
    ----------
    start_date, end_date:
        Inclusive "YYYY-MM-DD" range. All required datastreams must already
        be downloaded locally (scripts/download_nsa_data.py).
    include_extensions:
        Also average MET (2-m temperature) and QCRAD (broadband fluxes) over
        each window when their files are present locally. Extension beyond
        the Hartig26 instrument set; missing files just produce NaN columns.
    skip_radar:
        Skip KAZR entirely (fast, for working on the light instruments while
        radar data are still downloading). Radar-derived variables are NaN.
    verbose:
        Print one progress line per ~20 soundings.

    Returns
    -------
    xr.Dataset keyed by launch_time; see module docstring for contents.
    """
    window = np.timedelta64(int(config.COORDINATION_WINDOW_S), "s")

    # --- anchor: soundings --------------------------------------------------
    sondes = read_sondes(start_date, end_date)
    layers = saturated_layer_summary(sondes)
    launch_times = sondes["launch_time"].values
    n_launch = launch_times.size
    if verbose:
        print(f"{n_launch} soundings anchor the library "
              f"({sondes.attrs.get('n_soundings_dropped', 0)} dropped)")

    # --- continuous light datastreams: read once ---------------------------
    # Pad the end by a day so the last launch's forward window is covered.
    padded_end = str(
        (dt.date.fromisoformat(end_date) + dt.timedelta(days=1)).isoformat()
    )

    def _try_read(reader, label: str) -> Optional[xr.Dataset]:
        try:
            return reader(start_date, padded_end)
        except FileNotFoundError:
            if verbose:
                print(f"note: no local {label} files; its columns will be NaN")
            return None

    mwr = _try_read(read_mwr, "MWR")
    ceil = _try_read(read_ceilometer, "ceilometer")
    met = _try_read(read_met, "MET") if include_extensions else None
    qcrad = _try_read(read_qcrad, "QCRAD") if include_extensions else None

    # --- per-launch accumulation --------------------------------------------
    radar_grid = radar_height_grid_m()
    scalars: Dict[str, List[float]] = {
        "lwp_g_m2": [], "n_mwr_samples": [],
        "pwv_cm": [],
        "first_cbh_m": [], "ceil_cloud_fraction": [],
        "iwp_g_m2": [], "radar_coverage_fraction": [], "clear_sky": [],
        "temp_2m_c": [], "lwdn_w_m2": [], "swdn_w_m2": [], "lwup_w_m2": [],
    }
    cloud_frac_profiles = np.full((n_launch, radar_grid.size), np.nan)

    for i, t_launch in enumerate(launch_times):
        t0 = t_launch
        t1 = t_launch + window

        lwp_mean, n_mwr = _window_mean(mwr, "lwp_g_m2", t0, t1)
        scalars["lwp_g_m2"].append(lwp_mean)
        scalars["n_mwr_samples"].append(n_mwr)
        scalars["pwv_cm"].append(_window_mean(mwr, "pwv_cm", t0, t1)[0])

        scalars["first_cbh_m"].append(_window_mean(ceil, "first_cbh_m", t0, t1)[0])
        if ceil is not None:
            detected = ceil["cloud_detected"].sel(time=slice(t0, t1))
            frac = float(detected.mean()) if detected.size > 0 else float("nan")
        else:
            frac = float("nan")
        scalars["ceil_cloud_fraction"].append(frac)

        scalars["temp_2m_c"].append(_window_mean(met, "temp_2m_c", t0, t1)[0])
        scalars["lwdn_w_m2"].append(_window_mean(qcrad, "lwdn_w_m2", t0, t1)[0])
        scalars["swdn_w_m2"].append(_window_mean(qcrad, "swdn_w_m2", t0, t1)[0])
        scalars["lwup_w_m2"].append(_window_mean(qcrad, "lwup_w_m2", t0, t1)[0])

        # --- radar: load just this window ----------------------------------
        iwp = coverage = float("nan")
        clear = False
        if not skip_radar:
            try:
                kazr_win = read_kazr(str(t0), str(t1))
            except FileNotFoundError:
                kazr_win = None
            if kazr_win is not None and kazr_win.sizes.get("time", 0) > 0:
                coverage = window_coverage_fraction(kazr_win)
                # Hartig26: keep the hourly radar average only when at least
                # half the samples in the window report reflectivity.
                if coverage >= config.RADAR_MIN_COVERAGE_FRACTION:
                    iwp = iwp_g_m2(kazr_win)
                    clear = clear_sky_flag(kazr_win)
                    cloud_frac_profiles[i, :] = cloud_fraction_profile(
                        kazr_win
                    ).values
        scalars["iwp_g_m2"].append(iwp)
        scalars["radar_coverage_fraction"].append(coverage)
        scalars["clear_sky"].append(clear)

        if verbose and (i + 1) % 20 == 0:
            print(f"  coordinated {i + 1}/{n_launch} soundings")

    # --- assemble ------------------------------------------------------------
    out = xr.merge([sondes, layers])
    for name, values in scalars.items():
        dtype = bool if name == "clear_sky" else float
        out[name] = ("launch_time", np.asarray(values, dtype=dtype))
    out["cloud_frac_profile"] = (
        ("launch_time", "radar_height"),
        cloud_frac_profiles,
    )
    out = out.assign_coords(radar_height=radar_grid)

    out["lwp_g_m2"].attrs.update(
        units="g/m^2", long_name="MWR liquid water path, 1-h mean after launch"
    )
    out["pwv_cm"].attrs.update(
        units="cm", long_name="MWR precipitable water vapor, 1-h mean"
    )
    out["first_cbh_m"].attrs.update(
        units="m", long_name="ceilometer lowest cloud base, 1-h mean"
    )
    out["ceil_cloud_fraction"].attrs.update(
        units="1", long_name="fraction of ceilometer samples with hydrometeors"
    )
    out["iwp_g_m2"].attrs.update(
        units="g/m^2",
        long_name="ice water path from hourly-mean KAZR reflectivity",
    )
    out["radar_coverage_fraction"].attrs.update(
        units="1", long_name="fraction of window with valid KAZR profiles"
    )
    out["clear_sky"].attrs.update(
        long_name="KAZR clear-sky flag (Hartig26 criterion)"
    )
    out["cloud_frac_profile"].attrs.update(
        units="1", long_name="KAZR hydrometeor fraction per height bin, 1-h window"
    )
    out["radar_height"].attrs.update(units="m", long_name="radar height above ground")
    out["temp_2m_c"].attrs.update(units="degC", long_name="MET 2-m temperature, 1-h mean")
    out["lwdn_w_m2"].attrs.update(units="W/m^2", long_name="downwelling LW, 1-h mean")
    out["swdn_w_m2"].attrs.update(units="W/m^2", long_name="downwelling SW, 1-h mean")
    out["lwup_w_m2"].attrs.update(units="W/m^2", long_name="upwelling LW, 1-h mean")

    out.attrs.update(
        title="Sonde-coordinated NSA multi-instrument library",
        methodology=(
            "Instruments averaged over the hour following each radiosonde "
            "launch, per Hartig et al. (2026) Sect. 2; radar means require "
            f">= {config.RADAR_MIN_COVERAGE_FRACTION:.0%} sample coverage."
        ),
        history=f"built {dt.datetime.now(dt.timezone.utc).isoformat()} by arm_nsa.coordinate",
        period=f"{start_date} .. {end_date}",
    )
    return out


def save_library(library: xr.Dataset, filename: Optional[str] = None) -> str:
    """Write the library to data/processed/ as netCDF; returns the path."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        t0 = pd.Timestamp(library["launch_time"].values.min()).strftime("%Y%m%d")
        t1 = pd.Timestamp(library["launch_time"].values.max()).strftime("%Y%m%d")
        filename = f"nsa_sonde_coordinated_library.{t0}_{t1}.nc"
    path = config.PROCESSED_DATA_DIR / filename
    library.to_netcdf(path)
    return str(path)
