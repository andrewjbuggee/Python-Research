#!/usr/bin/env python3
"""Classify every ERA5 grid cell, at every time step, into five surface types.

The categories
==============
Combines the time-INVARIANT land-sea mask (``lsm``, the land fraction of a grid
box, from ``download_era5_land_sea_mask.py``) with the time-VARYING sea ice
concentration (``siconc``) carried in the SEB files::

    land          lsm >= 1 - tol                       (no water in the box)
    coastal       tol < lsm < 1 - tol                  (box holds land AND water)
    open ocean    lsm <= tol  and  siconc <  0.05
    marginal ice  lsm <= tol  and  0.05 <= siconc <= 0.95
    sea ice       lsm <= tol  and  siconc >  0.95

Only the ocean split moves in time. Land and coastal are fixed by ``lsm`` alone,
so a cell's membership in those two is constant across the whole record; the
seasonal migration all happens between the last three.

Why this partition, and the ice-edge problem it solves
------------------------------------------------------
The latitude at which the surface freezes moves by several degrees from year to
year, so a fixed-latitude average smears open water, marginal ice, and pack ice
together in different proportions each season. Averaging within these classes
instead follows the ice edge rather than fighting it.

``tol`` (default 1e-4) is packing round-off insurance, not a physical threshold;
see ``analyze_land_sea_mask.py``.

ONE IMPORTANT DEPARTURE FROM THE OBVIOUS DEFINITION
===================================================
ERA5 does not store ``siconc = 0`` over land. It stores **NaN**: sea ice
concentration is undefined outside the ocean tile. A literal "land is lsm >= 1-tol
AND siconc < 0.001" test therefore matches NOTHING, because NaN fails every
comparison. Land here is accepted when ``siconc`` is either NaN (the normal case)
or genuinely below ``--land-max-siconc``. Any land cell carrying ice cover above
that is left UNCLASSIFIED and counted in the report rather than silently folded
in, since it would mean ``lsm`` and ERA5's own surface tiling disagree.

The report prints how many land cell-times had finite ``siconc``. Expect zero. A
non-zero count is worth understanding before trusting the classification.

Unclassified cells
------------------
A cell is UNCLASSIFIED (code -1) only when it is pure sea by ``lsm`` but has no
finite ``siconc``, or is pure land with ice cover above the land tolerance.
Neither should occur; both are reported so they cannot pass unnoticed.

Coastal cells take precedence over any ice test. A box that is 40% land is
coastal whether or not its ocean fraction is frozen, because the SEB of a mixed
box is a blend that none of the three ocean categories describes.

Area weighting
==============
Every share this module reports is cos(latitude)-weighted, matching
``seb_analysis_common.area_weights``. On a regular lat/lon grid a 70N cell covers
roughly twice the surface of an 80N one, so a raw cell count would overstate the
poleward end of the domain.

Usage
=====
As a library::

    from surface_classification import classify_cells, load_land_sea_mask

As a script, to report the class breakdown of a downloaded region::

    ./surface_classification.py --region barrow
    ./surface_classification.py --region barrow --start 2021-08-01 --end 2022-03-31
    ./surface_classification.py --region barrow --monthly

Requires the mask from ``download_era5_land_sea_mask.py`` to exist for the region.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from download_era5_land_sea_mask import LSM_SHORT_NAME, mask_path
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    parse_date,
    resolve_data_root,
    resolve_region_dir,
)

# ----------------------------------------------------------------------------
# The five classes
# ----------------------------------------------------------------------------
UNCLASSIFIED = -1

CLASS_ORDER: tuple[str, ...] = (
    "land",
    "coastal",
    "open_ocean",
    "marginal_ice",
    "sea_ice",
)
CLASS_CODES: dict[str, int] = {name: i for i, name in enumerate(CLASS_ORDER)}

CLASS_LABELS: dict[str, str] = {
    "land": "Land",
    "coastal": "Coastal (mixed)",
    "open_ocean": "Open ocean",
    "marginal_ice": "Marginal ice zone",
    "sea_ice": "Sea ice",
}

# Chosen for separability in both hue and lightness, so the five lines stay
# distinguishable in greyscale and to a red-green colour-blind reader: brown,
# orange, dark blue, mid cyan, light grey.
CLASS_COLORS: dict[str, str] = {
    "land": "#8c6d4f",
    "coastal": "#e08214",
    "open_ocean": "#1f5fa8",
    "marginal_ice": "#4eb3d3",
    "sea_ice": "#9aa5ad",
}

# Packing round-off tolerance on lsm, not a physical threshold.
DEFAULT_LSM_TOL = 0.01

# Sea ice concentration cuts separating the three ocean classes.
DEFAULT_OPEN_OCEAN_MAX_SICONC = 0.05
DEFAULT_SEA_ICE_MIN_SICONC = 0.95

# Ice cover a pure-land cell may carry and still count as land. ERA5 stores NaN
# there, so in practice this only catches a disagreement between lsm and ERA5's
# surface tiling.
DEFAULT_LAND_MAX_SICONC = 0.001

# Time steps loaded into memory at once. The full grid over a long record does
# not fit; classification is streamed in blocks so peak memory stays bounded
# regardless of region size or record length.
DEFAULT_BLOCK_HOURS = 24 * 30


# ----------------------------------------------------------------------------
# Mask loading and alignment
# ----------------------------------------------------------------------------
def strip_frequency_suffix(region: str) -> str:
    """``barrow_daily`` -> ``barrow``; the mask is frequency-independent.

    One mask serves every temporal resolution of a region, since ``lsm`` has no
    time axis at all.
    """
    for suffix in ("_daily", "_monthly"):
        if region.endswith(suffix):
            return region[: -len(suffix)]
    return region


def load_land_sea_mask(region: str, data_root: Path, grid_deg: float | None = None):
    """Open the land-sea mask for ``region`` as a 2-D DataArray of land fraction."""
    import xarray as xr

    base_region = strip_frequency_suffix(region)
    path = mask_path(Path(data_root), base_region, grid_deg)
    if not path.exists():
        raise FileNotFoundError(
            f"No land-sea mask at {path}\n"
            f"  Download it first:\n"
            f"    ./download_era5_land_sea_mask.py --region {base_region}"
            + (f" --grid {grid_deg}" if grid_deg is not None else "")
        )
    with xr.open_dataset(path) as ds:
        if LSM_SHORT_NAME not in ds:
            raise KeyError(f"{path.name} has no {LSM_SHORT_NAME!r} variable")
        da = ds[LSM_SHORT_NAME].load()
    if "valid_time" in da.dims:  # a mask saved before the squeeze
        da = da.isel(valid_time=0, drop=True)
    return da


def align_lsm_to_grid(lsm_da, ds) -> np.ndarray:
    """Reindex the mask onto the data's own lat/lon and return it as (lat, lon).

    Reindexing by coordinate LABEL rather than trusting array order, because a
    mask and a data file that happen to store latitude in opposite directions
    would otherwise be combined upside down -- a silent error that would put the
    ice edge in the wrong place rather than raising.
    """
    aligned = lsm_da.reindex(
        latitude=ds["latitude"], longitude=ds["longitude"],
        method="nearest", tolerance=1e-4,
    )
    values = aligned.values.astype(np.float64)
    if np.isnan(values).any():
        n_missing = int(np.isnan(values).sum())
        raise ValueError(
            f"The land-sea mask does not cover {n_missing:,} of the data's "
            f"{values.size:,} grid cells.\n"
            f"  mask   lat {float(lsm_da['latitude'].min()):.2f} to "
            f"{float(lsm_da['latitude'].max()):.2f}, "
            f"lon {float(lsm_da['longitude'].min()):.2f} to "
            f"{float(lsm_da['longitude'].max()):.2f}\n"
            f"  data   lat {float(ds['latitude'].min()):.2f} to "
            f"{float(ds['latitude'].max()):.2f}, "
            f"lon {float(ds['longitude'].min()):.2f} to "
            f"{float(ds['longitude'].max()):.2f}\n"
            f"  The mask was probably downloaded for a different region or grid."
        )
    return values


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------
def classify_cells(
    lsm: np.ndarray,
    siconc: np.ndarray,
    lsm_tol: float = DEFAULT_LSM_TOL,
    open_ocean_max_siconc: float = DEFAULT_OPEN_OCEAN_MAX_SICONC,
    sea_ice_min_siconc: float = DEFAULT_SEA_ICE_MIN_SICONC,
    land_max_siconc: float = DEFAULT_LAND_MAX_SICONC,
) -> np.ndarray:
    """Assign a class code to every cell-time.

    Parameters
    ----------
    lsm : Land fraction, shape ``(n_lat, n_lon)``, on [0, 1]. Time-invariant.
    siconc : Sea ice cover, shape ``(..., n_lat, n_lon)``, on [0, 1], NaN over
        land. Broadcast against ``lsm``.

    Returns
    -------
    ndarray of int8 with ``siconc``'s shape, holding the codes in
    ``CLASS_CODES``, or ``UNCLASSIFIED`` (-1).
    """
    if lsm.shape != siconc.shape[-2:]:
        raise ValueError(
            f"lsm shape {lsm.shape} does not match the trailing spatial "
            f"dimensions of siconc {siconc.shape[-2:]}"
        )

    is_pure_land = lsm >= 1.0 - lsm_tol
    is_pure_sea = lsm <= lsm_tol
    is_coastal = ~(is_pure_land | is_pure_sea)

    ice_defined = np.isfinite(siconc)
    # NaN comparisons are all False, so every ice test below is implicitly
    # restricted to cells where siconc is defined. That is the intent.
    with np.errstate(invalid="ignore"):
        below_open = siconc < open_ocean_max_siconc
        above_pack = siconc > sea_ice_min_siconc
        land_ice_ok = siconc < land_max_siconc

    out = np.full(np.broadcast_shapes(siconc.shape, lsm.shape), UNCLASSIFIED,
                  dtype=np.int8)

    # Land: ice undefined (the normal ERA5 state) or negligible.
    out = np.where(is_pure_land & (~ice_defined | land_ice_ok),
                   CLASS_CODES["land"], out)
    # Coastal wins outright: a mixed box is not describable by an ocean class.
    out = np.where(is_coastal, CLASS_CODES["coastal"], out)
    # The three ocean classes, all requiring a defined siconc.
    sea = is_pure_sea & ice_defined
    out = np.where(sea & below_open, CLASS_CODES["open_ocean"], out)
    out = np.where(sea & ~below_open & ~above_pack, CLASS_CODES["marginal_ice"], out)
    out = np.where(sea & above_pack, CLASS_CODES["sea_ice"], out)
    return out


def class_weight_sums(classes: np.ndarray, weights_2d: np.ndarray) -> np.ndarray:
    """Total area weight in each class, per time step.

    Parameters
    ----------
    classes : ``(n_time, n_lat, n_lon)`` codes from ``classify_cells``.
    weights_2d : ``(n_lat, n_lon)`` cos(lat) weights.

    Returns
    -------
    ``(n_time, len(CLASS_ORDER))`` of summed weights.
    """
    n_time = classes.shape[0]
    out = np.zeros((n_time, len(CLASS_ORDER)))
    for name, code in CLASS_CODES.items():
        out[:, code] = ((classes == code) * weights_2d).sum(axis=(1, 2))
    return out


def area_weights_2d(lat_deg: np.ndarray, n_lon: int) -> np.ndarray:
    """cos(latitude) weights broadcast to the (lat, lon) grid."""
    return np.repeat(np.cos(np.deg2rad(lat_deg))[:, None], n_lon, axis=1)


def iter_time_blocks(
    ds,
    var_names: list[str],
    block_hours: int = DEFAULT_BLOCK_HOURS,
    keep_mask: np.ndarray | None = None,
):
    """Yield ``(start_index, loaded_block)`` over the time axis in slices.

    The whole record at full spatial resolution does not fit in memory for a
    long pull, so every consumer of this module streams. Only ``var_names`` are
    materialised, which is the difference between tens of megabytes and tens of
    gigabytes when the file carries 35 variables.

    ``keep_mask`` is a per-time-step boolean of what the caller will actually
    use. Blocks it excludes entirely are skipped WITHOUT being read, which is
    what makes asking for one season out of a 26-year archive cost one season of
    I/O rather than twenty-six.
    """
    n_time = ds.sizes["valid_time"]
    for i0 in range(0, n_time, block_hours):
        i1 = min(i0 + block_hours, n_time)
        if keep_mask is not None and not keep_mask[i0:i1].any():
            continue
        yield i0, ds[var_names].isel(valid_time=slice(i0, i1)).load()


# ----------------------------------------------------------------------------
# CLI report
# ----------------------------------------------------------------------------
def add_classification_args(parser: argparse.ArgumentParser) -> None:
    """The classification thresholds, shared with the plotting scripts."""
    group = parser.add_argument_group("surface classification")
    group.add_argument(
        "--lsm-tol", type=float, default=DEFAULT_LSM_TOL, metavar="TOL",
        help=(
            "How close lsm must be to 0 or 1 to count as pure sea or pure land. "
            f"Packing round-off insurance, not physics (default {DEFAULT_LSM_TOL:g})."
        ),
    )
    group.add_argument(
        "--open-ocean-max-siconc", type=float,
        default=DEFAULT_OPEN_OCEAN_MAX_SICONC, metavar="F",
        help=f"Sea ice below this is open ocean (default {DEFAULT_OPEN_OCEAN_MAX_SICONC:g}).",
    )
    group.add_argument(
        "--sea-ice-min-siconc", type=float,
        default=DEFAULT_SEA_ICE_MIN_SICONC, metavar="F",
        help=f"Sea ice above this is pack ice (default {DEFAULT_SEA_ICE_MIN_SICONC:g}). "
             "Between the two cuts is the marginal ice zone.",
    )
    group.add_argument(
        "--land-max-siconc", type=float,
        default=DEFAULT_LAND_MAX_SICONC, metavar="F",
        help=(
            "Ice cover a pure-land cell may carry and still count as land "
            f"(default {DEFAULT_LAND_MAX_SICONC:g}). ERA5 stores NaN over land, "
            "which is accepted regardless; this only catches a disagreement "
            "between lsm and ERA5's surface tiling."
        ),
    )
    group.add_argument(
        "--mask-grid", type=float, default=None, metavar="DEG",
        help="Read the mask saved at this regridded resolution instead of native.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    add_classification_args(parser)
    parser.add_argument("--start", type=parse_date, default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, default=None, metavar="YYYY-MM-DD",
                        help="Inclusive.")
    parser.add_argument("--monthly", action="store_true",
                        help="Also break the area shares down by calendar month.")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 72)
    print("Surface classification of ERA5 grid cells")
    print("=" * 72)

    try:
        region_dir = resolve_region_dir(args)
        ds = load_seb_data(args.region, args.start, args.end, region_dir.parent)
        lsm_da = load_land_sea_mask(
            args.region, resolve_data_root(args.storage, args.data_root), args.mask_grid
        )
        lsm = align_lsm_to_grid(lsm_da, ds)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    if "siconc" not in ds.data_vars:
        print("  Error: dataset has no 'siconc'; re-download with --var-set "
              "recommended or extended.", file=sys.stderr)
        return 1

    times = ds["valid_time"].values
    lat_deg = ds["latitude"].values
    weights = area_weights_2d(lat_deg, ds.sizes["longitude"])
    w_domain = float(weights.sum())

    print(f"  Source      : {region_dir}")
    print(f"  Grid        : {ds.sizes['latitude']} x {ds.sizes['longitude']} cells")
    print(f"  Period      : {str(times[0])[:13]} to {str(times[-1])[:13]} "
          f"({times.size:,} time steps)")
    print(f"  Thresholds  : lsm tol {args.lsm_tol:g} | open ocean siconc < "
          f"{args.open_ocean_max_siconc:g} | pack ice > {args.sea_ice_min_siconc:g}")
    print("-" * 72)

    n_class = len(CLASS_ORDER)
    w_by_time = np.zeros((times.size, n_class))
    n_unclassified = 0
    n_land_with_ice = 0

    for i0, block in iter_time_blocks(ds, ["siconc"], args.block_hours):
        siconc = block["siconc"].values
        classes = classify_cells(
            lsm, siconc, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )
        w_by_time[i0 : i0 + classes.shape[0]] = class_weight_sums(classes, weights)
        n_unclassified += int((classes == UNCLASSIFIED).sum())
        # The diagnostic promised in the module docstring: land cells whose
        # siconc is finite, which should never happen.
        n_land_with_ice += int(
            ((lsm >= 1.0 - args.lsm_tol) & np.isfinite(siconc)).sum()
        )

    share = 100.0 * w_by_time / w_domain     # (time, class), % of domain area

    print(f"  {'class':<20}{'mean %':>10}{'min %':>10}{'max %':>10}"
          f"{'time-invariant':>16}")
    print("  " + "-" * 66)
    for name in CLASS_ORDER:
        c = CLASS_CODES[name]
        s = share[:, c]
        fixed = "yes" if np.ptp(s) < 1e-9 else "no"
        print(f"  {CLASS_LABELS[name]:<20}{s.mean():>10.2f}{s.min():>10.2f}"
              f"{s.max():>10.2f}{fixed:>16}")
    print("  " + "-" * 66)
    print(f"  {'total':<20}{share.sum(axis=1).mean():>10.2f}")

    print()
    n_cell_times = times.size * lsm.size
    print(f"  Unclassified cell-times : {n_unclassified:,} of {n_cell_times:,} "
          f"({100.0 * n_unclassified / n_cell_times:.4f}%)")
    print(f"  Land cell-times with finite siconc : {n_land_with_ice:,} "
          f"(expected 0; ERA5 leaves siconc undefined over land)")
    if n_unclassified:
        print("    !! Non-zero. These are pure-sea cells with no siconc, or land "
              "cells carrying ice above --land-max-siconc.", file=sys.stderr)

    if args.monthly:
        print()
        print("  Area share by calendar month (%):")
        months = times.astype("datetime64[M]").astype(int) % 12 + 1
        header = "".join(f"{CLASS_LABELS[n][:9]:>11}" for n in CLASS_ORDER)
        print(f"    {'month':<7}{header}")
        import calendar as _cal
        for m in range(1, 13):
            sel = months == m
            if not sel.any():
                continue
            row = "".join(f"{share[sel, CLASS_CODES[n]].mean():>11.2f}"
                          for n in CLASS_ORDER)
            print(f"    {_cal.month_abbr[m]:<7}{row}")

    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
