#!/usr/bin/env python3
"""Count land-only, sea-only, and mixed grid cells in an ERA5 land-sea mask.

Reads what ``download_era5_land_sea_mask.py`` writes.

The three categories
====================
ERA5's ``lsm`` is the PROPORTION of a grid box covered by land, on [0, 1], so a
coastal box is genuinely fractional rather than binary::

    sea only    lsm <= tol           no land in the box
    land only   lsm >= 1 - tol       no water in the box
    mixed       everything between   a coastline runs through the box

``tol`` (``--tol``, default 1e-4) exists because the CDS netCDF backend has
historically packed fields as scaled 16-bit integers, which perturbs an exact 0
or 1 by ~1e-5. Testing ``lsm == 0`` would then classify true ocean as mixed. The
report prints how many cells are exactly 0 or 1 as well, so you can see whether
the tolerance is doing any work in your file.

Counts are reported three ways, because they answer different questions:

    raw cells         what the arrays contain; the right denominator for "how
                      many samples does my analysis have"
    area-weighted     cos(latitude) weighting, matching
                      ``seb_analysis_common.area_weights``. Grid cell area scales
                      as cos(lat), so a 70N cell covers ~2x an 80N cell and a raw
                      count over-represents the poleward end of the domain.
    land-fraction     sum(lsm * w) / sum(w), the actual fraction of the domain's
                      SURFACE that is land, which splits the mixed cells rather
                      than assigning them wholesale

A threshold sensitivity block then shows where the mixed cells go under the
0.5 cut ECMWF recommends for a binary decision, since that is the choice any
downstream masking has to make.

examples
========
  ./analyze_land_sea_mask.py
  ./analyze_land_sea_mask.py --region barrow --verbose
  ./analyze_land_sea_mask.py --file data/land_sea_masks/era5_lsm_barrow.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from download_era5_land_sea_mask import (
    DEFAULT_REGIONS,
    LSM_SHORT_NAME,
    mask_path,
)
from download_era5_seb import STORAGE_ROOTS, resolve_output_root
from era5_seb_variables import region_names

# Packing tolerance, not a physical threshold. See the module docstring.
DEFAULT_TOL = 1e-4

# ECMWF's recommended cut when a binary land/sea decision is unavoidable.
BINARY_THRESHOLD = 0.5


def classify(
    lsm: np.ndarray, tol: float = DEFAULT_TOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks ``(is_sea, is_land, is_mixed)`` over the mask array.

    The three are mutually exclusive and cover every finite cell, so their counts
    sum to the cell total.
    """
    is_sea = lsm <= tol
    is_land = lsm >= 1.0 - tol
    is_mixed = ~(is_sea | is_land)
    return is_sea, is_land, is_mixed


def area_weights_2d(lat_deg: np.ndarray, n_lon: int) -> np.ndarray:
    """cos(latitude) weights broadcast to the (lat, lon) grid.

    Same convention as ``seb_analysis_common.area_weights``: on a regular lat/lon
    grid, cell area is proportional to cos(lat) and independent of longitude.
    """
    return np.repeat(np.cos(np.deg2rad(lat_deg))[:, None], n_lon, axis=1)


def _pct(part: float, whole: float) -> float:
    """Percentage, with an empty domain reported as 0 rather than a NaN."""
    return 100.0 * part / whole if whole else 0.0


def report_one(path: Path, tol: float, verbose: bool) -> dict:
    """Print the full breakdown for one mask file and return its summary."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if LSM_SHORT_NAME not in ds:
            raise KeyError(
                f"{path.name} has no {LSM_SHORT_NAME!r}; got {sorted(ds.data_vars)}"
            )
        da = ds[LSM_SHORT_NAME]
        if "valid_time" in da.dims:  # tolerate a file saved before the squeeze
            da = da.isel(valid_time=0, drop=True)
        lsm = da.values.astype(np.float64)
        lat_deg = ds["latitude"].values.astype(np.float64)
        lon_deg = ds["longitude"].values.astype(np.float64)
        region_label = ds.attrs.get("region", path.stem)
        grid_attr = ds.attrs.get("grid_deg", "unknown")

    n_lat, n_lon = lsm.shape
    n_total = lsm.size

    n_nan = int(np.isnan(lsm).sum())
    if n_nan:
        # Not expected: lsm is defined everywhere, including over the pole.
        print(f"  [!] {n_nan} NaN cells in {path.name}; excluded from all counts")
        lsm = np.where(np.isnan(lsm), np.nan, lsm)

    finite = ~np.isnan(lsm)
    is_sea, is_land, is_mixed = classify(lsm, tol)
    is_sea &= finite
    is_land &= finite
    is_mixed &= finite

    weights = area_weights_2d(lat_deg, n_lon)
    w_total = float(weights[finite].sum())
    w_sea = float(weights[is_sea].sum())
    w_land = float(weights[is_land].sum())
    w_mixed = float(weights[is_mixed].sum())

    n_sea, n_land, n_mixed = int(is_sea.sum()), int(is_land.sum()), int(is_mixed.sum())
    n_finite = int(finite.sum())

    # True surface land fraction: splits each mixed cell by its own value rather
    # than assigning the whole cell to one category.
    land_area_fraction = (
        float(np.nansum(lsm * weights)) / w_total if w_total else 0.0
    )

    print("=" * 72)
    print(f"{region_label}  ({path.name})")
    print("=" * 72)
    print(f"  Grid          : {n_lat} lat x {n_lon} lon = {n_total:,} cells "
          f"at {grid_attr} deg")
    print(f"  Latitude      : {lat_deg.min():.2f} to {lat_deg.max():.2f} N")
    print(f"  Longitude     : {lon_deg.min():.2f} to {lon_deg.max():.2f} E")
    print(f"  lsm range     : {np.nanmin(lsm):.6f} to {np.nanmax(lsm):.6f}")
    print(f"  Tolerance     : {tol:g} (cells within this of 0 or 1 count as pure)")
    print()
    print(f"  {'category':<14}{'cells':>10}{'% cells':>10}"
          f"{'% area':>10}   (area = cos-lat weighted)")
    print("  " + "-" * 62)
    print(f"  {'sea only':<14}{n_sea:>10,}{_pct(n_sea, n_finite):>9.2f}%"
          f"{_pct(w_sea, w_total):>9.2f}%")
    print(f"  {'land only':<14}{n_land:>10,}{_pct(n_land, n_finite):>9.2f}%"
          f"{_pct(w_land, w_total):>9.2f}%")
    print(f"  {'mixed':<14}{n_mixed:>10,}{_pct(n_mixed, n_finite):>9.2f}%"
          f"{_pct(w_mixed, w_total):>9.2f}%")
    print("  " + "-" * 62)
    print(f"  {'total':<14}{n_finite:>10,}{_pct(n_finite, n_finite):>9.2f}%"
          f"{_pct(w_total, w_total):>9.2f}%")
    print()
    print(f"  Land fraction of the domain surface : {100 * land_area_fraction:.2f}%")
    print("    (sum(lsm*w)/sum(w) -- splits mixed cells by their own value)")

    # Is the tolerance actually doing anything, or is the file already exact?
    n_exact_0 = int((lsm[finite] == 0.0).sum())
    n_exact_1 = int((lsm[finite] == 1.0).sum())
    print(f"  Exactly 0.0 / 1.0                   : {n_exact_0:,} / {n_exact_1:,}")
    if n_exact_0 != n_sea or n_exact_1 != n_land:
        print(f"    -> the tolerance reclassified "
              f"{(n_sea - n_exact_0) + (n_land - n_exact_1):,} near-pure cells; "
              f"values are stored inexactly (packed), as expected")

    if n_mixed:
        mixed_values = lsm[is_mixed]
        n_mostly_sea = int((mixed_values < BINARY_THRESHOLD).sum())
        n_mostly_land = n_mixed - n_mostly_sea
        print()
        print(f"  Mixed cells ({n_mixed:,}), land fraction within them:")
        print(f"    min / median / max : {mixed_values.min():.4f} / "
              f"{np.median(mixed_values):.4f} / {mixed_values.max():.4f}")
        print(f"    mean               : {mixed_values.mean():.4f}")
        edges = np.linspace(0.0, 1.0, 11)
        counts, _ = np.histogram(mixed_values, bins=edges)
        print("    distribution (bin -> cells):")
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            bar = "#" * int(round(40 * count / max(counts.max(), 1)))
            print(f"      {lo:.1f}-{hi:.1f} {count:>6,}  {bar}")
        print()
        print(f"  Under a {BINARY_THRESHOLD} threshold (ECMWF's binary recommendation):")
        print(f"    mixed -> sea  : {n_mostly_sea:,}"
              f"   giving {n_sea + n_mostly_sea:,} sea cells "
              f"({_pct(n_sea + n_mostly_sea, n_finite):.2f}%)")
        print(f"    mixed -> land : {n_mostly_land:,}"
              f"   giving {n_land + n_mostly_land:,} land cells "
              f"({_pct(n_land + n_mostly_land, n_finite):.2f}%)")

    if verbose and n_mixed:
        print()
        print("  Mixed cells by latitude band:")
        band_edges = np.arange(
            np.floor(lat_deg.min()), np.ceil(lat_deg.max()) + 1.0, 1.0
        )
        lat_of_cell = np.repeat(lat_deg[:, None], n_lon, axis=1)
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            in_band = (lat_of_cell >= lo) & (lat_of_cell < hi)
            n_band_mixed = int((is_mixed & in_band).sum())
            if n_band_mixed:
                n_band = int((finite & in_band).sum())
                print(f"    {lo:5.1f}-{hi:5.1f}N : {n_band_mixed:>6,} mixed "
                      f"of {n_band:,} ({_pct(n_band_mixed, n_band):.2f}%)")

    print()
    return {
        "region": region_label,
        "n_total": n_finite,
        "n_sea": n_sea,
        "n_land": n_land,
        "n_mixed": n_mixed,
        "land_area_fraction": land_area_fraction,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_land_sea_mask.py",
        description=(
            "Count land-only, sea-only, and mixed (coastal) grid cells in an "
            "ERA5 land-sea mask downloaded by download_era5_land_sea_mask.py. "
            "Reports raw counts, cos(lat) area-weighted shares, and the true "
            "land fraction of the domain surface."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--region",
        nargs="+",
        choices=region_names() + ["custom_area"],
        default=list(DEFAULT_REGIONS),
        metavar="NAME",
        help=f"Region mask(s) to read. (default: {' '.join(DEFAULT_REGIONS)})",
    )
    parser.add_argument(
        "--file",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help="Explicit mask file(s), bypassing --region/--storage entirely.",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=None,
        metavar="DEG",
        help="Read the mask saved at this regridded resolution instead of native.",
    )
    parser.add_argument(
        "--storage",
        choices=sorted(STORAGE_ROOTS),
        default="local",
        help="Which root the masks were written to. (default: local)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="PATH",
        help="Explicit directory holding the mask files, overriding --storage.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=(
            "How close to 0 or 1 still counts as pure sea or pure land. Guards "
            f"against integer-packing round-off. (default: {DEFAULT_TOL:g})"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also break the mixed cells down by 1-degree latitude band.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.file is not None:
        paths = [Path(p).expanduser() for p in args.file]
    else:
        root = resolve_output_root(args.storage, args.data_root)
        paths = [
            mask_path(root, name, args.grid) for name in dict.fromkeys(args.region)
        ]

    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Missing mask file(s):", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print("\nDownload them first:", file=sys.stderr)
        print("  ./download_era5_land_sea_mask.py --region "
              + " ".join(dict.fromkeys(args.region)), file=sys.stderr)
        return 1

    summaries = [report_one(p, args.tol, args.verbose) for p in paths]

    if len(summaries) > 1:
        print("=" * 72)
        print("summary")
        print("=" * 72)
        print(f"  {'region':<18}{'cells':>10}{'sea':>10}{'land':>10}"
              f"{'mixed':>10}{'land area':>11}")
        for s in summaries:
            print(f"  {s['region']:<18}{s['n_total']:>10,}{s['n_sea']:>10,}"
                  f"{s['n_land']:>10,}{s['n_mixed']:>10,}"
                  f"{100 * s['land_area_fraction']:>10.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
