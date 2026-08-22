#!/usr/bin/env python3
"""Download the ERA5 land-sea mask (``lsm``) once per region.

Why this is a separate script
=============================
``lsm`` is one of ERA5's INVARIANT fields: it is written at every timestep of the
archive but its value never changes, so a single retrieval describes the whole
1940-present record. Bundling it into ``download_era5_seb.py`` would re-download
an identical 2-D field alongside every hourly chunk. For the Barrow strip that is
2,501 cells x 4 bytes on every one of ~9,500 files, which buys nothing.

The companion script ``analyze_land_sea_mask.py`` reads what this writes and
counts land / sea / mixed cells.

What the field actually is
==========================
``lsm`` is the PROPORTION of each grid box covered by land, on [0, 1], not a
binary flag. Cells that straddle a coastline take intermediate values, which is
why the counting script has three categories rather than two. ECMWF's own
recommendation is to threshold at 0.5 when a binary land/sea decision is needed
(ERA5 documentation, "Parameter listings / land-sea mask").

Two caveats worth carrying into any analysis built on this:

* Inland water. ERA5 handles lakes through a separate ``lake_cover`` (``cl``)
  field within the land tile, so a lake generally does NOT appear as sea in
  ``lsm``. Moderate confidence -- verify against ``cl`` if inland water matters
  for your domain. It does not for the Barrow strip, which is coastal ocean.
* Regridding. Passing ``--grid`` makes the CDS interpolate the mask, which
  manufactures new intermediate values along coastlines that the native 0.25 deg
  field does not have. The output filename records the grid for that reason.

Relationship to the existing land mask
======================================
``seb_analysis_common.build_ocean_mask`` currently identifies land as "``siconc``
is NaN", which is a hard binary test applied by ERA5's own surface tiling. That
is not the same partition as ``lsm``: a cell that is 60% land still carries a
defined ``siconc`` for its ocean portion. Comparing the two is a good sanity
check before switching any analysis over.

Output
======
One file per region, written under ``<root>/land_sea_masks/`` rather than in a
region subdirectory::

    <root>/land_sea_masks/era5_lsm_barrow.nc
    <root>/land_sea_masks/era5_lsm_arctic_circle_0p50deg.nc

Kept out of the ``<root>/<region>/`` trees deliberately: those hold time-series
chunks that the analysis scripts glob and concatenate, and mask files with no
time axis in there would be a foot-gun. ``land_sea_masks`` itself is kept out of
``seb_analysis_common.available_regions`` by name (see ``LSM_SUBDIR`` above), so
it is never listed or loaded as if it were a region.

The stored field has its length-1 ``valid_time`` dimension squeezed out, so it
broadcasts against ``(valid_time, latitude, longitude)`` data without fuss.

Requires ``cdsapi`` and the same ``~/.cdsapirc`` as ``download_era5_seb.py``.

examples
========
  # both default regions, to the local data/ directory
  ./download_era5_land_sea_mask.py

  # show the plan without contacting the CDS
  ./download_era5_land_sea_mask.py --dry-run

  # one region, external drive, matching a coarsened data pull
  ./download_era5_land_sea_mask.py --region arctic_circle --grid 0.5 \
      --storage external
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from download_era5_seb import (
    DEFAULT_CDS_RETRIES,
    NATIVE_GRID_DEG,
    STORAGE_ROOTS,
    check_volume_mounted,
    download_one,
    resolve_output_root,
    verify_credentials,
)
from era5_seb_variables import Region, get_region, region_names

# The invariant field lives in the hourly single-level archive; there is no
# dedicated "invariants" dataset on the current CDS.
DATASET = "reanalysis-era5-single-levels"

LSM_CDS_NAME = "land_sea_mask"
LSM_SHORT_NAME = "lsm"

# Subdirectory of the storage root the masks live in. Named here once and
# imported by seb_analysis_common.available_regions, which excludes it by this
# same name -- otherwise a directory of masks with no time axis would get
# listed as if it were a region a script could load SEB data from.
LSM_SUBDIR = "land_sea_masks"

# Any timestamp returns the same field. This one is arbitrary, chosen only to sit
# well inside the final-ERA5 period rather than the preliminary ERA5T window near
# real time, so nobody has to wonder whether the mask came from a provisional
# stream.
SNAPSHOT_DAY = date(2020, 1, 1)
SNAPSHOT_HOUR = "00:00"

# 'arctic' in conversation means the pan-Arctic box the downloader also defaults
# to. Both regions the SEB record has been pulled for are covered here.
DEFAULT_REGIONS: tuple[str, ...] = ("barrow", "arctic_circle")


def grid_tag(grid_deg: float | None) -> str:
    """Filename fragment recording the grid, empty for the native resolution.

    A regridded mask is a genuinely different field from the native one -- the
    interpolation invents new coastal fractions -- so the two must never collide
    on one filename.
    """
    if grid_deg is None:
        return ""
    return "_" + f"{grid_deg:.2f}".replace(".", "p") + "deg"


def mask_path(out_root: Path, region_label: str, grid_deg: float | None) -> Path:
    """Where the mask for one region lands."""
    return out_root / LSM_SUBDIR / f"era5_lsm_{region_label}{grid_tag(grid_deg)}.nc"


def build_lsm_request(area_deg: list[float], grid_deg: float | None) -> dict:
    """One CDS request for a single timestep of the land-sea mask.

    Parameters
    ----------
    area_deg : ``[North, West, South, East]`` in degrees, CDS ordering.
    grid_deg : Regrid resolution, or None for the native 0.25 deg.
    """
    request = {
        "product_type": ["reanalysis"],
        "variable": [LSM_CDS_NAME],
        "year": [f"{SNAPSHOT_DAY.year:04d}"],
        "month": [f"{SNAPSHOT_DAY.month:02d}"],
        "day": [f"{SNAPSHOT_DAY.day:02d}"],
        "time": [SNAPSHOT_HOUR],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area_deg,
    }
    if grid_deg is not None:
        request["grid"] = [grid_deg, grid_deg]
    return request


def finalise_mask(
    path: Path, region_label: str, region: Region, grid_deg: float | None
) -> tuple[int, int]:
    """Drop the length-1 time axis and stamp provenance onto the file.

    Returns ``(n_lat, n_lon)``. Rewriting through a scratch path rather than in
    place keeps a failure here from leaving a truncated mask that a later run
    would treat as complete.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        out = ds.load()

    if LSM_SHORT_NAME not in out:
        raise KeyError(
            f"{path.name} has no {LSM_SHORT_NAME!r} variable; got "
            f"{sorted(out.data_vars)}. The CDS response was not the land-sea mask."
        )

    # squeeze(drop=True) rather than isel: this asserts the axis really is
    # length 1. A mask that came back with several timesteps means the request
    # was not what this script thinks it was, and should fail loudly.
    if "valid_time" in out.dims:
        if out.sizes["valid_time"] != 1:
            raise ValueError(
                f"{path.name} has {out.sizes['valid_time']} timesteps; the "
                f"land-sea mask request should return exactly one."
            )
        out = out.squeeze("valid_time", drop=True)

    out.attrs.update(
        {
            "title": f"ERA5 land-sea mask, region {region_label}",
            "region": region_label,
            "region_description": region.description,
            "area_north_west_south_east_deg": str(region.as_area()),
            "grid_deg": grid_deg if grid_deg is not None else NATIVE_GRID_DEG,
            "regridded": "no" if grid_deg is None else "yes (CDS interpolation)",
            "source_dataset": DATASET,
            "snapshot_time_utc": f"{SNAPSHOT_DAY.isoformat()} {SNAPSHOT_HOUR}",
            "invariant_note": (
                "lsm is time-invariant in ERA5; this single timestep describes "
                "the whole archive. The time axis has been squeezed out."
            ),
            "lsm_definition": "proportion of the grid box covered by land, 0-1",
            "created_by": Path(__file__).name,
        }
    )

    tmp_path = path.with_suffix(".final.part")
    out.to_netcdf(
        tmp_path, encoding={LSM_SHORT_NAME: {"zlib": True, "complevel": 4}}
    )
    tmp_path.replace(path)
    return out.sizes["latitude"], out.sizes["longitude"]


def download_region_mask(
    client,
    region_label: str,
    region: Region,
    out_root: Path,
    grid_deg: float | None,
    overwrite: bool,
) -> bool:
    """Retrieve, consolidate, and finalise the mask for one region."""
    out_path = mask_path(out_root, region_label, grid_deg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {region_label:<18} -> {out_path.name}")

    if out_path.exists() and not overwrite:
        size_kb = out_path.stat().st_size / 1024
        print(f"    already present ({size_kb:.1f} KB); --overwrite to replace")
        return True

    request = build_lsm_request(region.as_area(), grid_deg)
    ok, size_mb = download_one(client, DATASET, request, out_path)
    if not ok:
        print(f"    FAILED for region {region_label}", file=sys.stderr)
        return False

    try:
        n_lat, n_lon = finalise_mask(out_path, region_label, region, grid_deg)
    except (KeyError, ValueError, OSError) as exc:
        print(f"    consolidated but could not finalise: {exc}", file=sys.stderr)
        return False

    print(
        f"    written: {n_lat} x {n_lon} = {n_lat * n_lon:,} cells, "
        f"{out_path.stat().st_size / 1024:.1f} KB "
        f"(from a {size_mb:.2f} MB payload)"
    )
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_era5_land_sea_mask.py",
        description=(
            "Download the time-invariant ERA5 land-sea mask (lsm) once per "
            "region, using the same named boxes as download_era5_seb.py. The "
            "field is the fraction of each grid box covered by land, on [0, 1]."
        ),
        epilog=(
            "examples:\n"
            "  download_era5_land_sea_mask.py --dry-run\n"
            "  download_era5_land_sea_mask.py --region barrow\n"
            "  download_era5_land_sea_mask.py --region arctic_circle --grid 0.5\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--region",
        nargs="+",
        choices=region_names(),
        default=list(DEFAULT_REGIONS),
        metavar="NAME",
        help=(
            "One or more named boxes from era5_seb_variables.REGIONS: "
            f"{', '.join(region_names())}. "
            f"(default: {' '.join(DEFAULT_REGIONS)})"
        ),
    )
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=None,
        help=(
            "Ad hoc box in degrees, overriding --region. CDS ordering: North "
            "West South East, longitude on [-180, 180]. Output is labelled "
            "'custom_area'."
        ),
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Regrid to a coarser resolution, to match a coarsened data pull. "
            "Omit for the native 0.25 deg. Interpolation invents new coastal "
            "fractions, so the resolution is recorded in the filename."
        ),
    )
    parser.add_argument(
        "--storage",
        choices=sorted(STORAGE_ROOTS),
        default="local",
        help=(
            "Which preconfigured root to write under, same as the SEB "
            "downloader. (default: local)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help="Write somewhere else entirely, ignoring --storage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download masks whose file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and the request without contacting the CDS.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.area is not None:
        north_deg, west_deg, south_deg, east_deg = args.area
        if north_deg <= south_deg:
            raise SystemExit(
                f"--area NORTH ({north_deg}) must exceed SOUTH ({south_deg})."
            )
        targets = [
            ("custom_area", Region(north_deg, west_deg, south_deg, east_deg,
                                   "explicit --area"))
        ]
    else:
        # dict.fromkeys: de-duplicate a repeated --region while keeping order.
        targets = [(name, get_region(name)) for name in dict.fromkeys(args.region)]

    out_root = resolve_output_root(args.storage, args.out_dir)

    print("=" * 72)
    print("ERA5 land-sea mask download (time-invariant field)")
    print("=" * 72)
    print(f"  Dataset      : {DATASET}")
    print(f"  Variable     : {LSM_CDS_NAME} ({LSM_SHORT_NAME}), land fraction 0-1")
    print(f"  Snapshot     : {SNAPSHOT_DAY} {SNAPSHOT_HOUR} UTC (value is time-invariant)")
    print(f"  Grid         : {args.grid or NATIVE_GRID_DEG} deg")
    print(f"  Output root  : {out_root}")
    print("-" * 72)
    for label, region in targets:
        print(f"  {label:<18} {region.as_area()}  -- {region.description}")
    print("-" * 72)

    if args.dry_run:
        example_label, example_region = targets[0]
        print("Dry run; nothing was downloaded.")
        print(f"\nRequest that would be sent for {example_label}:")
        for key, value in build_lsm_request(example_region.as_area(), args.grid).items():
            print(f"    {key:<18} {value}")
        print("\nFiles that would be written:")
        for label, _ in targets:
            print(f"    {mask_path(out_root, label, args.grid)}")
        return 0

    if not verify_credentials():
        return 1

    check_volume_mounted(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    import cdsapi

    client = cdsapi.Client(retry_max=DEFAULT_CDS_RETRIES)

    n_failed = 0
    for label, region in targets:
        if not download_region_mask(
            client, label, region, out_root, args.grid, args.overwrite
        ):
            n_failed += 1

    print("-" * 72)
    if n_failed:
        print(f"{n_failed} of {len(targets)} region(s) failed.", file=sys.stderr)
        return 1
    print(f"All {len(targets)} region mask(s) present under {out_root}")
    print("Next: ./analyze_land_sea_mask.py --region " +
          " ".join(label for label, _ in targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
