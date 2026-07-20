"""KAZR cloud radar (Ka-band, zenith) processing.

Implements the radar treatment of Hartig26 Sect. 2:

* General ("ge") mode reflectivity, native ~4-5 s x 30 m resolution.
* Profiles are put on a shared height grid, 105 m to 12,000 m in 30-m steps.
* Gates with signal-to-noise ratio below -13 dB are treated as "no detection".
* Derived, after time averaging over a window (Hartig26 averages the hour
  following each sonde launch):
    - vertically resolved cloud (hydrometeor) fraction,
    - ice water path from IWC = a * Ze**b (a = 0.1 from SHEBA winter, Shupe et
      al. 2005; b = 0.63 from Matrosov 1999), Ze in linear units [mm^6 m^-3],
    - a clear-sky flag (>= 99% of gates below 10 km undetected and no
      contiguous detected region deeper than 100 m).

Note on radar "cloud" fraction: reflectivity responds to ALL hydrometeors, so
this fraction includes precipitation as well as cloud (Hartig26 makes the same
caveat). The IWC-Ze relation assumes ice-dominated returns, appropriate for
Arctic extended winter; in liquid drizzle it will misattribute mass.

Averaging-order caveat: Hartig26 states IWP is computed "after the hourly
averaging". Because IWC ~ Ze**0.63 is nonlinear, averaging Ze first then
applying the power law (default here, matching the paper's wording) is not
identical to averaging per-sample IWC; iwp_g_m2() exposes both via
`average_first`. Differences are modest but reproducibility demands the same
choice throughout an analysis.

KAZR files are large (one file per hour or few hours; ~10^2 MB each). Read
only the window you need -- the coordination module loads radar data one
sonde-hour at a time for exactly this reason.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

from . import config
from .readers import files_in_range, resolve_variable

_REFL_CANDIDATES = ("reflectivity", "reflectivity_copol")
_SNR_CANDIDATES = (
    "signal_to_noise_ratio",
    "signal_to_noise_ratio_copol",
    "snr",
    "snr_copol",
)
_RANGE_CANDIDATES = ("range",)


def radar_height_grid_m() -> np.ndarray:
    """Shared vertical grid for radar reflectivity [m above ground]."""
    return np.arange(
        config.RADAR_GRID_BOTTOM_M,
        config.RADAR_GRID_TOP_M + config.RADAR_GRID_STEP_M / 2,
        config.RADAR_GRID_STEP_M,
    )


def _read_one_kazr_file(path, grid_m: np.ndarray) -> Optional[xr.Dataset]:
    """Read one KAZR file onto the common grid with the SNR screen applied."""
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        refl_name = resolve_variable(ds, _REFL_CANDIDATES)
        range_name = resolve_variable(ds, _RANGE_CANDIDATES)
        refl_dbz = ds[refl_name].astype("float32")

        # "The radar reported at this time" must be distinguishable from "the
        # radar detected nothing": a clear hour and a radar outage otherwise
        # look identical after SNR screening. Record data presence BEFORE the
        # screen (raw files carry noise-level values at undetected gates).
        has_data = np.isfinite(refl_dbz).any(dim=range_name)

        # Screen by SNR when the file provides it; otherwise rely on the
        # ingest's own masking (non-detections arrive as NaN/_FillValue).
        try:
            snr_name = resolve_variable(ds, _SNR_CANDIDATES)
            snr_db = ds[snr_name]
            refl_dbz = refl_dbz.where(snr_db >= config.RADAR_MIN_SNR_DB)
        except KeyError:
            pass

        # Align onto the shared grid along the range dimension. The native
        # spacing is already ~30 m, so nearest-gate reindexing (tolerance one
        # grid step) is an alignment, not a resampling -- and it avoids
        # linearly blending reflectivities in logarithmic dBZ space.
        refl_dbz = refl_dbz.rename({range_name: "height"})
        refl_dbz = refl_dbz.assign_coords(height=ds[range_name].values)
        gridded = refl_dbz.reindex(
            height=grid_m, method="nearest", tolerance=config.RADAR_GRID_STEP_M
        )

        out = xr.Dataset(
            {"reflectivity_dbz": gridded, "has_data": has_data}
        )
        out["reflectivity_dbz"].attrs.update(
            units="dBZ",
            long_name="equivalent reflectivity factor (SNR-screened)",
            source_variable=refl_name,
        )
        out["has_data"].attrs.update(
            long_name="radar reported this time sample (pre-screening)"
        )
        out.attrs["source_file"] = path.name
        return out.load()


def read_kazr(
    start: str,
    end: str,
    verbose: bool = False,
) -> xr.Dataset:
    """Read KAZR reflectivity for [start, end] onto the common height grid.

    Parameters
    ----------
    start, end:
        Anything np.datetime64/xarray slicing understands, e.g. "2022-01-05"
        or "2022-01-05T12:00:00". Keep windows short (hours to a few days):
        full-resolution KAZR is voluminous.

    Returns
    -------
    Dataset with reflectivity_dbz(time, height); NaN marks "no detection"
    (below SNR threshold or missing). Also provides boolean `detected`
    (per gate) and `has_data` (per time: radar reported at all -- this is
    what separates a clear sky from a radar outage).
    """
    grid_m = radar_height_grid_m()
    start_day = str(np.datetime64(start, "D"))
    end_day = str(np.datetime64(end, "D"))
    paths = files_in_range("kazr", start_day, end_day)
    if not paths:
        raise FileNotFoundError(
            f"No local KAZR files for {start_day}..{end_day} under "
            f"{config.RAW_DATA_DIR}. Run scripts/download_nsa_data.py first "
            f"(datastream key 'kazr')."
        )
    pieces: List[xr.Dataset] = []
    for path in paths:
        if verbose:
            print(f"reading {path.name}")
        piece = _read_one_kazr_file(path, grid_m)
        if piece is not None:
            pieces.append(piece)

    combined = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
    combined = combined.sortby("time").sel(time=slice(start, end))
    combined["detected"] = np.isfinite(combined["reflectivity_dbz"])
    combined["detected"].attrs.update(
        long_name="hydrometeor detected (reflectivity above SNR threshold)"
    )
    combined["height"].attrs.update(units="m", long_name="height above ground")
    return combined


# ---------------------------------------------------------------------------
# Derived quantities (computed on a time window, e.g. one sonde-hour)
# ---------------------------------------------------------------------------


def ze_linear_mm6_m3(reflectivity_dbz: xr.DataArray) -> xr.DataArray:
    """Convert reflectivity from dBZ to linear Ze [mm^6 m^-3]."""
    ze = 10.0 ** (reflectivity_dbz / 10.0)
    ze.attrs.update(units="mm^6 m^-3", long_name="equivalent reflectivity factor")
    return ze


def _reporting_mask(kazr_window: xr.Dataset) -> xr.DataArray:
    """Boolean over time: the radar reported data at this sample.

    Uses the pre-screening `has_data` flag when present; falls back to
    "any finite screened reflectivity" for datasets built without it (that
    fallback cannot distinguish clear sky from outage -- see read_kazr).
    """
    if "has_data" in kazr_window:
        return kazr_window["has_data"]
    return np.isfinite(kazr_window["reflectivity_dbz"]).any(dim="height")


def cloud_fraction_profile(kazr_window: xr.Dataset) -> xr.DataArray:
    """Fraction of window time samples with detection, per height bin.

    Missing profiles (radar not reporting) are excluded from the denominator,
    mirroring "missing data is ignored in the hourly average" (Hartig26).
    """
    detected = kazr_window["detected"]
    valid_profile = _reporting_mask(kazr_window)
    n_valid = int(valid_profile.sum())
    if n_valid == 0:
        frac = xr.full_like(kazr_window["height"], np.nan, dtype=float)
    else:
        frac = detected.isel(time=np.where(valid_profile.values)[0]).mean(dim="time")
    frac.attrs.update(
        long_name="hydrometeor fraction (cloud + precipitation)", units="1"
    )
    return frac


def window_coverage_fraction(kazr_window: xr.Dataset) -> float:
    """Fraction of time samples in the window where the radar reported data.

    Hartig26 keeps an hourly radar average only when >= 50% of samples have
    measurements (config.RADAR_MIN_COVERAGE_FRACTION).
    """
    if kazr_window.sizes.get("time", 0) == 0:
        return 0.0
    return float(_reporting_mask(kazr_window).mean())


def iwp_g_m2(
    kazr_window: xr.Dataset,
    average_first: bool = True,
) -> float:
    """Ice water path [g m^-2] for a time window of KAZR data.

    IWC = a * Ze**b with Ze in mm^6 m^-3, IWC in g m^-3
    (a = config.IWC_PREFACTOR_A, b = config.IWC_EXPONENT_B), then integrated
    over height: IWP = sum(IWC * dz).

    Parameters
    ----------
    average_first:
        True  -> average linear Ze over time first, then apply the power law
                 (matches the Hartig26 wording "calculated after the hourly
                 averaging"). Non-detections count as Ze = 0.
        False -> apply the power law per sample, then time-average the IWC.

    Returns NaN when the radar reported nothing in the window; returns 0.0
    for a reporting radar with no detections (a clear window holds no ice).
    """
    refl_dbz = kazr_window["reflectivity_dbz"]
    if window_coverage_fraction(kazr_window) == 0.0:
        return float("nan")
    if not np.isfinite(refl_dbz).any():
        return 0.0

    ze_lin = ze_linear_mm6_m3(refl_dbz)
    # Non-detections contribute zero reflectivity/ice to the average.
    ze_filled = ze_lin.fillna(0.0)
    dz_m = config.RADAR_GRID_STEP_M

    if average_first:
        ze_mean = ze_filled.mean(dim="time")
        iwc_g_m3 = config.IWC_PREFACTOR_A * ze_mean**config.IWC_EXPONENT_B
        # A bin with zero mean reflectivity holds no ice; 0**0.63 = 0 already.
        return float((iwc_g_m3 * dz_m).sum())
    iwc_g_m3 = config.IWC_PREFACTOR_A * ze_filled**config.IWC_EXPONENT_B
    iwp_per_sample = (iwc_g_m3 * dz_m).sum(dim="height")
    return float(iwp_per_sample.mean())


def clear_sky_flag(kazr_window: xr.Dataset) -> bool:
    """Hartig26 clear-sky criterion for a time window of KAZR data.

    Clear when, in the window-averaged detection profile below 10 km:
      * at least 99% of bins show no detection at all, AND
      * no contiguous run of detecting bins is deeper than 100 m
        (on the 30-m grid, runs of up to 3 bins = 90 m are tolerated).

    Returns False when the window has no valid data (cannot claim clear).
    """
    if window_coverage_fraction(kazr_window) == 0.0:
        return False
    frac = cloud_fraction_profile(kazr_window)
    below = frac.sel(height=slice(None, config.CLEAR_SKY_MAX_HEIGHT_M))
    bin_detected = (below > 0).values

    clear_fraction = 1.0 - bin_detected.mean()
    if clear_fraction < config.CLEAR_SKY_MIN_CLEAR_FRACTION:
        return False

    # Longest contiguous run of detecting bins.
    max_run = run = 0
    for flag in bin_detected:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)
    max_depth_m = max_run * config.RADAR_GRID_STEP_M
    return max_depth_m <= config.CLEAR_SKY_MAX_LAYER_DEPTH_M
