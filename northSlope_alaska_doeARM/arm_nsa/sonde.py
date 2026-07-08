"""Radiosonde (SONDEWNPN) processing.

Implements the sounding treatment of Hartig26 Sect. 2:

* Each launch (up to 4/day) is interpolated onto a shared height grid running
  8 m to 12,000 m in 5-m steps.
* Launches that do not report data up to at least 1,000 m are discarded.
* "Saturated layers" -- the sounding-based proxy for cloud liquid regions --
  are contiguous regions with RH (w.r.t. liquid) >= 95%, after (a) filling
  sub-threshold gaps no deeper than 30 m that are bounded by saturated air,
  and (b) discarding layers thinner than 30 m.

Height convention: the sonde `alt` variable is altitude above mean sea level.
NSA C1 sits at ~8 m MSL (config.SITE_ALT_M), so the 8-m grid bottom is the
surface; heights here are effectively height above ground to within a few
meters. The Vaisala RH is reported with respect to liquid water at all
temperatures (standard radiosonde convention), which is exactly what the
saturated-layer definition requires -- no ice/liquid conversion is needed.

Radiosonde files hold one profile per file, so this module reads file-by-file
rather than concatenating along a continuous time axis like readers.py does.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import config
from .qc import apply_qc
from .readers import files_in_range, resolve_variable

# Variables interpolated onto the common grid (canonical name -> candidates).
_PROFILE_VARS = {
    "pres_hpa": ("pres",),
    "tdry_c": ("tdry", "temp"),
    "dp_c": ("dp",),
    "rh_pct": ("rh",),
    "wspd_m_s": ("wspd",),
    "u_wind_m_s": ("u_wind",),
    "v_wind_m_s": ("v_wind",),
}


def sonde_height_grid_m() -> np.ndarray:
    """The shared vertical grid for interpolated soundings [m MSL]."""
    return np.arange(
        config.SONDE_GRID_BOTTOM_M,
        config.SONDE_GRID_TOP_M + config.SONDE_GRID_STEP_M / 2,
        config.SONDE_GRID_STEP_M,
    )


def _interp_profile(
    alt_m: np.ndarray, values: np.ndarray, grid_m: np.ndarray
) -> np.ndarray:
    """Linearly interpolate one sounding variable onto the common grid.

    Grid points outside the sounding's altitude span are NaN (no
    extrapolation: a sounding that bursts at 9 km says nothing about 11 km).
    """
    good = np.isfinite(alt_m) & np.isfinite(values)
    if good.sum() < 2:
        return np.full(grid_m.shape, np.nan)
    alt_good = alt_m[good]
    val_good = values[good]
    # np.interp requires strictly increasing x; sondes can report brief
    # descents (pendulum motion) or duplicate altitudes.
    order = np.argsort(alt_good, kind="stable")
    alt_sorted = alt_good[order]
    val_sorted = val_good[order]
    keep = np.concatenate(([True], np.diff(alt_sorted) > 0))
    alt_sorted, val_sorted = alt_sorted[keep], val_sorted[keep]
    interped = np.interp(grid_m, alt_sorted, val_sorted)
    interped[(grid_m < alt_sorted[0]) | (grid_m > alt_sorted[-1])] = np.nan
    return interped


def read_sondes(
    start_date: str,
    end_date: str,
    apply_qc_flags: bool = True,
    verbose: bool = False,
) -> xr.Dataset:
    """Read and grid every valid sounding launched in [start_date, end_date].

    Returns
    -------
    Dataset with dims (launch_time, height):
        tdry_c, dp_c, rh_pct, pres_hpa, wspd_m_s, u_wind_m_s, v_wind_m_s
    `height` is m MSL on the common 5-m grid. Launch time is the time of the
    first record in each file (the balloon release).

    Notes
    -----
    Launches whose maximum reporting altitude is below config.SONDE_MIN_TOP_M
    (1,000 m) are dropped, following Hartig26. With 2011-2023 extended winters
    they report 4,791 valid soundings; expect similar counts if reproducing.
    """
    grid_m = sonde_height_grid_m()
    paths = files_in_range("sonde", start_date, end_date)
    if not paths:
        raise FileNotFoundError(
            f"No local sonde files in {start_date}..{end_date} under "
            f"{config.RAW_DATA_DIR}. Run scripts/download_nsa_data.py first."
        )

    launch_times: List[np.datetime64] = []
    profiles: List[dict] = []
    n_dropped = 0

    for path in paths:
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            alt_name = resolve_variable(ds, ("alt",))
            alt_m = np.asarray(ds[alt_name].values, dtype=float)
            if not np.isfinite(alt_m).any():
                n_dropped += 1
                continue
            if np.nanmax(alt_m) < config.SONDE_MIN_TOP_M:
                n_dropped += 1  # burst / failed early, unusable (Hartig26)
                continue

            launch_time = ds["time"].values[0]
            profile: dict = {}
            for canonical, candidates in _PROFILE_VARS.items():
                try:
                    native = resolve_variable(ds, candidates)
                except KeyError:
                    profile[canonical] = np.full(grid_m.shape, np.nan)
                    continue
                if apply_qc_flags:
                    values = apply_qc(ds, native).values.astype(float)
                else:
                    values = np.asarray(ds[native].values, dtype=float)
                profile[canonical] = _interp_profile(alt_m, values, grid_m)

        launch_times.append(launch_time)
        profiles.append(profile)
        if verbose:
            print(f"sonde {path.name}: gridded")

    if not profiles:
        raise RuntimeError(
            f"All {n_dropped} soundings in {start_date}..{end_date} were "
            f"unusable (no altitude data or top below "
            f"{config.SONDE_MIN_TOP_M:.0f} m)."
        )

    data_vars = {
        canonical: (
            ("launch_time", "height"),
            np.vstack([p[canonical] for p in profiles]),
        )
        for canonical in _PROFILE_VARS
    }
    out = xr.Dataset(
        data_vars,
        coords={
            "launch_time": np.array(launch_times, dtype="datetime64[ns]"),
            "height": grid_m,
        },
    )
    out = out.sortby("launch_time")
    out["height"].attrs.update(units="m", long_name="height above mean sea level")
    out["tdry_c"].attrs.update(units="degC", long_name="dry-bulb temperature")
    out["dp_c"].attrs.update(units="degC", long_name="dewpoint temperature")
    out["rh_pct"].attrs.update(
        units="%", long_name="relative humidity w.r.t. liquid water"
    )
    out["pres_hpa"].attrs.update(units="hPa", long_name="air pressure")
    out["wspd_m_s"].attrs.update(units="m/s", long_name="wind speed")
    out["u_wind_m_s"].attrs.update(units="m/s", long_name="eastward wind")
    out["v_wind_m_s"].attrs.update(units="m/s", long_name="northward wind")
    out.attrs["n_soundings_dropped"] = n_dropped
    return out


# ---------------------------------------------------------------------------
# Saturated layers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SaturatedLayer:
    """One contiguous saturated layer in a single profile."""

    base_m: float  # height of lowest saturated grid cell
    top_m: float  # height of highest saturated grid cell
    depth_m: float  # cell-inclusive depth: top - base + one grid step


def find_saturated_layers(
    rh_pct: np.ndarray,
    height_m: np.ndarray,
    rh_threshold_pct: float = config.SATURATION_RH_THRESHOLD_PCT,
    max_gap_m: float = config.SATURATED_LAYER_MAX_GAP_M,
    min_depth_m: float = config.SATURATED_LAYER_MIN_DEPTH_M,
) -> List[SaturatedLayer]:
    """Locate saturated layers in one RH profile (Hartig26 Sect. 2).

    Algorithm
    ---------
    1. Flag grid cells with RH >= threshold (NaN counts as unsaturated).
    2. Fill any unsaturated gap of depth <= max_gap_m that is bounded above
       and below by saturated cells (small RH dips cannot split a layer).
    3. Group the remaining flags into contiguous runs; discard runs with
       cell-inclusive depth < min_depth_m.

    Depth convention: a run spanning cells i..j on a grid with step dz has
    depth (j - i + 1) * dz, i.e. each cell contributes one grid step. On the
    5-m sonde grid a 30-m minimum therefore requires >= 6 consecutive cells.

    Parameters
    ----------
    rh_pct:
        1-D relative humidity w.r.t. liquid [%] on `height_m`.
    height_m:
        1-D monotonically increasing, uniformly spaced heights [m].

    Returns
    -------
    List of SaturatedLayer, ordered bottom to top.
    """
    rh = np.asarray(rh_pct, dtype=float)
    z = np.asarray(height_m, dtype=float)
    if rh.shape != z.shape:
        raise ValueError("rh_pct and height_m must have identical shapes")
    dz = float(np.median(np.diff(z)))

    saturated = np.where(np.isfinite(rh), rh >= rh_threshold_pct, False)
    if not saturated.any():
        return []

    # --- step 2: fill small bounded gaps -----------------------------------
    # Runs of consecutive equal values: run starts where the flag changes.
    change_idx = np.flatnonzero(np.diff(saturated.astype(np.int8)))
    run_starts = np.concatenate(([0], change_idx + 1))
    run_ends = np.concatenate((change_idx, [saturated.size - 1]))
    filled = saturated.copy()
    for start, end in zip(run_starts, run_ends):
        if saturated[start]:
            continue  # a saturated run, nothing to fill
        gap_depth_m = (end - start + 1) * dz
        bounded = start > 0 and end < saturated.size - 1
        if bounded and gap_depth_m <= max_gap_m:
            filled[start : end + 1] = True

    # --- step 3: extract runs and apply the minimum depth ------------------
    change_idx = np.flatnonzero(np.diff(filled.astype(np.int8)))
    run_starts = np.concatenate(([0], change_idx + 1))
    run_ends = np.concatenate((change_idx, [filled.size - 1]))
    layers: List[SaturatedLayer] = []
    for start, end in zip(run_starts, run_ends):
        if not filled[start]:
            continue
        depth_m = (end - start + 1) * dz
        if depth_m >= min_depth_m:
            layers.append(
                SaturatedLayer(base_m=float(z[start]), top_m=float(z[end]), depth_m=depth_m)
            )
    return layers


def saturated_layer_summary(sondes: xr.Dataset) -> xr.Dataset:
    """Per-launch saturated-layer statistics for a gridded sonde Dataset.

    Parameters
    ----------
    sondes:
        Output of read_sondes() -- needs rh_pct(launch_time, height).

    Returns
    -------
    Dataset over launch_time with:
        n_saturated_layers        count
        total_saturated_depth_m   sum of layer depths [m]
        lowest_base_m             base of lowest layer [m MSL], NaN if none
        lowest_top_m              top of lowest layer [m MSL], NaN if none
        highest_top_m             top of highest layer [m MSL], NaN if none
    """
    z = sondes["height"].values
    n_layers = []
    total_depth_m = []
    lowest_base_m = []
    lowest_top_m = []
    highest_top_m = []

    for i in range(sondes.sizes["launch_time"]):
        layers = find_saturated_layers(sondes["rh_pct"].isel(launch_time=i).values, z)
        n_layers.append(len(layers))
        total_depth_m.append(sum(layer.depth_m for layer in layers))
        lowest_base_m.append(layers[0].base_m if layers else np.nan)
        lowest_top_m.append(layers[0].top_m if layers else np.nan)
        highest_top_m.append(layers[-1].top_m if layers else np.nan)

    return xr.Dataset(
        {
            "n_saturated_layers": ("launch_time", np.array(n_layers, dtype=int)),
            "total_saturated_depth_m": ("launch_time", np.array(total_depth_m)),
            "lowest_base_m": ("launch_time", np.array(lowest_base_m)),
            "lowest_top_m": ("launch_time", np.array(lowest_top_m)),
            "highest_top_m": ("launch_time", np.array(highest_top_m)),
        },
        coords={"launch_time": sondes["launch_time"]},
    )
