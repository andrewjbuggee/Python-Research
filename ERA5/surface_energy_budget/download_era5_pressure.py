#!/usr/bin/env python3
"""Download ERA5 hourly data on PRESSURE LEVELS, for cloud water content.

Sibling of ``download_era5_seb.py``, which downloads the single-level archive.
Same region definitions, same storage layout, same resume behaviour, same retry
policy -- all of that machinery is imported rather than copied, so a fix in one
place fixes both. What differs is the dataset, the variables, and one thing that
changes the arithmetic completely: the pressure-level axis.

THE LEVEL AXIS MULTIPLIES THE COST, AND IT DOMINATES EVERYTHING
==============================================================
The CDS charges a request by how much hourly ERA5 it must touch:

    fields = variables x LEVELS x days x 24

The single-level downloader has no level term. Here it is the largest factor in
the product. With 8 variables on all 37 levels, ONE day costs

    8 x 37 x 24 = 7,104 fields

against a measured hourly ceiling of 12,648 -- so a single request could not
even hold two days. Trimming to the troposphere (23 levels) brings a day to
4,416 fields and lets two days share a request; dropping to the lower
troposphere (16 levels) gets four days in.

Two levers, in order of how much they buy you for what they cost:

  --levels     Cuts cost proportionally and, above 200 hPa in this region,
               costs no information at all: there is no liquid above the winter
               tropopause. This is the lever to pull.
  --var-set    minimal (6 vars) is 25% cheaper than extended (8). See
               era5_pressure_variables.py for what each set can and cannot do.

The default is --levels troposphere --var-set standard: 23 levels x 7
variables, 3,864 fields per day, 3 days per request, 81 requests for a season.

Geopotential and relative humidity are deliberately NOT in that set. Both are
derivable from what is -- z by integrating the hypsometric equation upward from
the surface using t and q, r from t, q and the level pressure -- and dropping
them is what takes a request from 2 days to 3, and the season from 124 requests
to 81. Since queue time dominates, that is a 33% cut in wall-clock time for no
loss of information. --var-set extended or requested adds them back.

WHAT COMES BACK, AND WHAT TO DO WITH IT
=======================================
Condensate arrives as SPECIFIC content, kg per kg of moist air. Turning that
into a mass path or a density is what ``convert_specific_to_absolute.py`` does;
the physics and the required inputs are documented in
``era5_pressure_variables.py``. In short: a mass path needs only the specific
content and the pressure coordinate, a density needs temperature and specific
humidity as well, and an in-cloud value needs the cloud fraction.

Pressure levels below the ground are extrapolated, not measured. Mask them with
the single-level ``sp`` you already have before integrating anything.

SIZE AND TIME
=============
A 0.25 deg Barrow strip is 41 x 61 = 2,501 cells. One variable, one level, one
hour is 2,501 floats = 10 kB. So an Aug 1 - Mar 31 season (243 days) at 8
variables x 23 levels is

    7 x 23 x 24 x 243 x 2501 x 4 bytes ~= 9.4 GB

before compression, roughly 4-5 GB written. That is 7x what the same season
costs on single levels, and it will take proportionally longer. Start it and
leave it; --resume means an interrupted run picks up where it stopped.

OPTIONS
=======

Data source and destination
---------------------------
--storage {local,external}   Which disk to write to (default external).
--out-dir PATH               Explicit directory, overriding --storage.
--region NAME                Named region from era5_seb_variables.REGIONS.
--area N W S E               Explicit bounding box, overriding --region.

Time span
---------
--start YYYY-MM-DD           First day, inclusive (default 2025-08-01).
--end YYYY-MM-DD             Last day, inclusive (default 2026-03-31).
--chunk-days N               Days per request (default: computed from the
                             variable and level counts to stay under the
                             measured field ceiling).

What to download
----------------
--var-set NAME               minimal | standard | extended | requested.
--levels SPEC                troposphere | lower | deep | all, or an explicit
                             list ('1000,925,850') or range ('1000-500').
--grid DEG                   Resample to a coarser grid. Cuts bytes, not cost.

Behaviour
---------
--jobs N                     Concurrent requests (default 1). The CDS runs ONE
                             request per user, so raising this does not speed
                             anything up -- it was measured, not assumed.
--cds-retries N              Connection retries inside cdsapi (default 10).
--overwrite                  Re-download days already on disk.
--dry-run                    Print the plan and the cost arithmetic, download
                             nothing.

Examples
--------
    # the default: one Barrow winter season, troposphere, 7 variables
    python download_era5_pressure.py --region barrow --storage local

    # see the plan and the per-request field count without downloading
    python download_era5_pressure.py --region barrow --dry-run

    # cheapest useful configuration
    python download_era5_pressure.py --region barrow \\
        --var-set minimal --levels lower
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

# Imported, not duplicated: region definitions, chunking, resume scanning,
# consolidation, retry policy, credential checks. Any fix there lands here too.
from download_era5_seb import (
    ALL_HOURS,
    DEFAULT_CDS_RETRIES,
    STORAGE_ROOTS,
    _thread_client,
    check_volume_mounted,
    chunk_by_month,
    contiguous_runs,
    daterange,
    download_one,
    scan_covered_days,
    split_evenly,
    verify_credentials,
)
from era5_pressure_variables import (
    DEFAULT_LEVEL_SET,
    DEFAULT_VAR_SET,
    VARIABLE_SETS,
    describe_set,
    resolve_levels,
    variables_for,
)
from era5_seb_variables import REGIONS

PRESSURE_DATASET = "reanalysis-era5-pressure-levels"

# MEASURED on this dataset. Every value here was observed, and the earlier
# number in this slot was not -- it was the single-level ceiling assumed to
# carry over, which it does NOT. That assumption cost a rejected chunk:
#
#   8 vars x 23 levels, 2 days    8,832 fields   ACCEPTED   (probe)
#   7 vars x 23 levels, 2 days    7,728 fields   ACCEPTED   (real run)
#   7 vars x 23 levels, 3 days   11,592 fields   REJECTED   "cost limits exceeded"
#   8 vars x 23 levels, 4 days   17,664 fields   REJECTED   (probe)
#
# So the pressure-level ceiling lies in (8,832, 11,592] -- BELOW the 12,648 the
# single-level archive accepts. The value below is the largest count verified to
# be accepted here, not an extrapolation from another dataset.
#
# It is a floor on the truth, not the truth. That is why the rejection is now
# also handled at run time: download_chunk_adaptive halves a chunk the CDS
# refuses and retries the halves, so a wrong constant costs a little time
# instead of silently losing days from the archive.
#
# Separately, and more important for wall clock: the accepted 2-day probe took
# 726 s over a 2x2-cell area against 6 s of transfer for the full strip. Queue
# time dominates completely and is independent of payload, so runtime is set by
# the NUMBER of requests. The ceiling caps how few of them there can be.
SAFE_REQUEST_FIELDS = 8_832

DEFAULT_START = date(2025, 8, 1)
DEFAULT_END = date(2026, 3, 31)
DEFAULT_REGION = "barrow"
DEFAULT_STORAGE = "external"

NATIVE_GRID_DEG = 0.25
BYTES_PER_VALUE = 4


# ----------------------------------------------------------------------------
# Cost arithmetic
# ----------------------------------------------------------------------------
def fields_per_day(n_vars: int, n_levels: int) -> int:
    """Hourly source fields one day of this request costs."""
    return n_vars * n_levels * 24


def max_chunk_days(n_vars: int, n_levels: int,
                   ceiling: int = SAFE_REQUEST_FIELDS) -> int:
    """Largest number of days that still fits under the field ceiling.

    Never returns 0: if even a single day exceeds the ceiling the request is
    still issued as one day, because there is no smaller unit to split into.
    The CDS may reject it, and the error handler says so explicitly rather than
    letting the caller guess.
    """
    per_day = fields_per_day(n_vars, n_levels)
    return max(1, ceiling // per_day)


def estimate_size_gb(n_vars: int, n_levels: int, n_days: int,
                     area: list[float], grid_deg: float) -> float:
    """Uncompressed size of the whole request, GB."""
    north, west, south, east = area
    n_lat = int(round(abs(north - south) / grid_deg)) + 1
    n_lon = int(round(abs(east - west) / grid_deg)) + 1
    values = n_vars * n_levels * 24 * n_days * n_lat * n_lon
    return values * BYTES_PER_VALUE / 1024**3


# ----------------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------------
def build_request(
    variables: list[str],
    levels: tuple[int, ...],
    area: list[float],
    year: int,
    month: int,
    days: list[date],
    grid_deg: float | None = None,
) -> dict:
    """One CDS request for the pressure-level hourly archive.

    Differs from the single-level request only by ``pressure_level``, which the
    CDS wants as strings.
    """
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "pressure_level": [str(p) for p in levels],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d.day:02d}" for d in days],
        "time": ALL_HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }
    if grid_deg is not None:
        request["grid"] = [grid_deg, grid_deg]
    return request


def plan_chunks(days: list[date], chunk_days: int) -> list[list[date]]:
    """Split the requested days into per-request chunks.

    Month-aligned first, because the CDS request takes one year and one month;
    then split to the day limit; then split around days already on disk so a
    resume never re-requests what it already has.
    """
    chunks: list[list[date]] = []
    for _year, _month, month_days in chunk_by_month(days):
        for run in contiguous_runs(month_days):
            chunks.extend(split_evenly(run, chunk_days))
    return chunks


def chunk_filename(region: str, chunk: list[date]) -> str:
    """Same naming shape the single-level downloader uses, so resume scanning
    (``days_covered_by_file``) recognises these files without changes."""
    first, last = chunk[0], chunk[-1]
    if first == last:
        return f"era5_pl_{region}_{first:%Y%m%d}.nc"
    return (f"era5_pl_{region}_{first:%Y%m}"
            f"_{first.day:02d}-{last.day:02d}.nc")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def check_existing_variables(run_dir: Path, wanted: list[str]) -> list[str]:
    """Warn if files already on disk hold a different variable set.

    Resume works in DATE space: a day already present is skipped, whatever it
    contains. That is right for an interrupted run and wrong after a --var-set
    change, because the skipped days keep the OLD variables and the new ones get
    the new set. The result opens fine -- xarray fills the gaps with NaN -- and
    is quietly inconsistent, which is the worst way for it to fail.

    Returns the variable short names found, or an empty list if nothing is on
    disk yet or the check could not be made.
    """
    existing = sorted(run_dir.glob("*.nc"))
    if not existing:
        return []
    try:
        import xarray as xr
        with xr.open_dataset(existing[0]) as ds:
            return sorted(str(v) for v in ds.data_vars)
    except (OSError, ValueError, ImportError):
        return []


# CDS variable name -> the short name it lands under in the netCDF.
_SHORT_NAME = {
    "specific_cloud_liquid_water_content": "clwc",
    "specific_cloud_ice_water_content": "ciwc",
    "specific_rain_water_content": "crwc",
    "specific_snow_water_content": "cswc",
    "temperature": "t",
    "specific_humidity": "q",
    "fraction_of_cloud_cover": "cc",
    "geopotential": "z",
    "relative_humidity": "r",
}


def download_chunk_adaptive(chunk, *, variables, levels, area, grid_deg,
                            run_dir, region_name, retries, max_retries,
                            depth=0):
    """Fetch one chunk, halving it and retrying if the CDS refuses it.

    The per-request ceiling is known only as a range (see SAFE_REQUEST_FIELDS),
    and it is a property of the CDS rather than of this code, so it can move.
    Treating a rejection as fatal loses those days from the archive silently --
    the run reports a failure count and carries on, and the gap only surfaces
    much later when something tries to read the season.

    Halving instead costs one wasted queue wait and converges in a couple of
    steps: a 3-day chunk that is refused becomes 2 + 1, both of which fit. A
    single day that is still refused is genuinely unfittable and is reported.

    Returns a list of ``(chunk, ok, mb)``, one per piece actually attempted.
    """
    out_path = run_dir / chunk_filename(region_name, chunk)
    request = build_request(variables, levels, area,
                            year=chunk[0].year, month=chunk[0].month,
                            days=chunk, grid_deg=grid_deg)
    client = _thread_client(retries)
    ok, mb = download_one(client, PRESSURE_DATASET, request, out_path,
                          max_retries=max_retries)
    if ok or len(chunk) == 1:
        return [(chunk, ok, mb)]

    mid = len(chunk) // 2
    pad = "  " * (depth + 1)
    print(f"    {pad}refused at {len(chunk)} day(s); splitting into "
          f"{mid} + {len(chunk) - mid}")
    results = []
    for half in (chunk[:mid], chunk[mid:]):
        results.extend(download_chunk_adaptive(
            half, variables=variables, levels=levels, area=area,
            grid_deg=grid_deg, run_dir=run_dir, region_name=region_name,
            retries=retries, max_retries=max_retries, depth=depth + 1))
    return results


def parse_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a date in YYYY-MM-DD form"
        ) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--storage", choices=sorted(STORAGE_ROOTS),
                   default=DEFAULT_STORAGE)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--region", default=DEFAULT_REGION,
                   help=f"Named region (default {DEFAULT_REGION}). "
                        f"Available: {', '.join(sorted(REGIONS))}")
    p.add_argument("--area", nargs=4, type=float, default=None,
                   metavar=("N", "W", "S", "E"))
    p.add_argument("--start", type=parse_date, default=DEFAULT_START)
    p.add_argument("--end", type=parse_date, default=DEFAULT_END)
    p.add_argument("--chunk-days", type=int, default=None,
                   help="Days per request. Default: the largest that fits "
                        "under the measured field ceiling.")
    p.add_argument("--var-set", choices=sorted(VARIABLE_SETS),
                   default=DEFAULT_VAR_SET)
    p.add_argument("--levels", default=DEFAULT_LEVEL_SET,
                   help="Level set name, list, or range (default "
                        f"{DEFAULT_LEVEL_SET}).")
    p.add_argument("--grid", type=float, default=None, metavar="DEG")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--cds-retries", type=int, default=DEFAULT_CDS_RETRIES)
    p.add_argument("--max-retries", type=int, default=4,
                   help="Per-chunk retries around a failed transfer.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.area is not None:
        area = list(args.area)
        region_name = "custom"
    else:
        if args.region not in REGIONS:
            print(f"  Error: unknown region {args.region!r}. "
                  f"Available: {', '.join(sorted(REGIONS))}", file=sys.stderr)
            return 1
        area = REGIONS[args.region].as_area()
        region_name = args.region

    try:
        levels = resolve_levels(args.levels)
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    variables = variables_for(args.var_set)

    n_vars, n_levels = len(variables), len(levels)
    per_day = fields_per_day(n_vars, n_levels)
    auto_chunk = max_chunk_days(n_vars, n_levels)
    chunk_days = args.chunk_days or auto_chunk

    try:
        days = daterange(args.start, args.end)
    except ValueError as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1

    root = args.out_dir or STORAGE_ROOTS[args.storage]
    run_dir = Path(root) / f"{region_name}_pressure"

    print("=" * 78)
    print("ERA5 pressure-level download")
    print("=" * 78)
    print(f"  Dataset    : {PRESSURE_DATASET}")
    print(f"  Region     : {region_name}  area N/W/S/E = {area}")
    print(f"  Span       : {args.start} to {args.end}  ({len(days)} days)")
    print(f"  Destination: {run_dir}")
    print(f"\n  Variables  : {args.var_set} ({n_vars})")
    print(describe_set(args.var_set))
    print(f"\n  Levels     : {args.levels} ({n_levels}) "
          f"{levels[0]}-{levels[-1]} hPa")
    print(f"    {', '.join(str(p) for p in levels)}")

    print(f"\n  Cost per day : {n_vars} vars x {n_levels} levels x 24 h "
          f"= {per_day:,} fields")
    print(f"  Field ceiling: {SAFE_REQUEST_FIELDS:,} per request")
    print(f"  Chunk size   : {chunk_days} day(s) "
          f"= {per_day * chunk_days:,} fields"
          + ("" if args.chunk_days is None else f"  (--chunk-days {chunk_days})"))
    if per_day * chunk_days > SAFE_REQUEST_FIELDS:
        print(f"  !! {per_day * chunk_days:,} exceeds the measured ceiling. The "
              f"CDS may reject these\n     requests. Reduce --levels, or "
              f"--var-set, or --chunk-days.", file=sys.stderr)

    grid_deg = args.grid or NATIVE_GRID_DEG
    total_gb = estimate_size_gb(n_vars, n_levels, len(days), area, grid_deg)
    print(f"  Size (raw)   : ~{total_gb:.1f} GB uncompressed over the whole span")

    if args.storage == "external" and args.out_dir is None:
        try:
            check_volume_mounted(Path(root))
        except (FileNotFoundError, OSError) as exc:
            print(f"\n  Error: {exc}", file=sys.stderr)
            return 1

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    on_disk = check_existing_variables(run_dir, variables)
    if on_disk:
        wanted_short = sorted(_SHORT_NAME.get(v, v) for v in variables)
        if set(on_disk) != set(wanted_short):
            only_disk = sorted(set(on_disk) - set(wanted_short))
            only_new = sorted(set(wanted_short) - set(on_disk))
            print(f"\n  !! The files already in {run_dir.name} hold a DIFFERENT "
                  f"variable set.", file=sys.stderr)
            print(f"     on disk : {', '.join(on_disk)}", file=sys.stderr)
            print(f"     wanted  : {', '.join(wanted_short)}", file=sys.stderr)
            if only_disk:
                print(f"     Resuming keeps {', '.join(only_disk)} on the old "
                      f"days and not the new ones.", file=sys.stderr)
            if only_new:
                print(f"     The old days will be MISSING "
                      f"{', '.join(only_new)}.", file=sys.stderr)
            print("     Either match the old set, or clear the directory "
                  "first:", file=sys.stderr)
            print(f"       rm {run_dir}/*.nc", file=sys.stderr)
            print("     Continuing would leave a mixed archive that opens "
                  "without error.\n", file=sys.stderr)

    have = set() if args.overwrite else scan_covered_days(run_dir)
    todo = [d for d in days if d not in have]
    if have:
        print(f"\n  Resume     : {len(have):,} day(s) already on disk, "
              f"{len(todo):,} to fetch")
    if not todo:
        print("\n  Nothing to do; every requested day is already present.")
        print("=" * 78)
        return 0

    chunks = plan_chunks(todo, chunk_days)
    print(f"  Requests   : {len(chunks)}")

    if args.dry_run:
        print("\n  Plan (first 10 of "
              f"{len(chunks)}):")
        for chunk in chunks[:10]:
            print(f"    {chunk_filename(region_name, chunk):<44}"
                  f"{len(chunk)} day(s)  {chunk[0]} .. {chunk[-1]}")
        if len(chunks) > 10:
            print(f"    ... and {len(chunks) - 10} more")
        print("\n  --dry-run: nothing downloaded.")
        print("=" * 78)
        return 0

    if not verify_credentials():
        return 1

    def work(chunk: list[date]):
        """One chunk, split further if the CDS refuses its size."""
        return download_chunk_adaptive(
            chunk, variables=variables, levels=levels, area=area,
            grid_deg=args.grid, run_dir=run_dir, region_name=region_name,
            retries=args.cds_retries, max_retries=args.max_retries)

    print()
    done = failed = 0
    total_mb = 0.0
    try:
        if args.jobs <= 1:
            for i, chunk in enumerate(chunks, 1):
                name = chunk_filename(region_name, chunk)
                print(f"  [{i}/{len(chunks)}] {name} ...")
                for piece, ok, mb in work(chunk):
                    done, failed = done + ok, failed + (not ok)
                    total_mb += mb
                    if ok:
                        print(f"      wrote {mb:,.1f} MB"
                              + (f"  ({chunk_filename(region_name, piece)})"
                                 if piece != chunk else ""))
                    else:
                        print(f"      FAILED {chunk_filename(region_name, piece)}")
        else:
            # Kept for completeness. The CDS runs one request per user, so this
            # does not speed the download up; it was measured on the
            # single-level archive and the result is in that script's README.
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = {pool.submit(work, c): c for c in chunks}
                for i, fut in enumerate(as_completed(futures), 1):
                    for piece, ok, mb in fut.result():
                        done, failed = done + ok, failed + (not ok)
                        total_mb += mb
                        print(f"  [{i}/{len(chunks)}] "
                              f"{chunk_filename(region_name, piece)} "
                              f"{'ok' if ok else 'FAILED'}  {mb:,.1f} MB")
    except KeyboardInterrupt:
        print("\n  Interrupted. Rerun to resume; finished chunks are kept.",
              file=sys.stderr)
        return 130

    print(f"\n  Done: {done} chunk(s), {total_mb/1024:.2f} GB written"
          + (f", {failed} FAILED" if failed else ""))
    if failed:
        print("  Rerun to retry the failures; completed days are skipped.")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
