#!/usr/bin/env python3
"""Download ERA5 single-level data for an Arctic surface energy budget (SEB)
analysis following Equation (1) of Sledd et al. (2025).

    Sledd, A., Shupe, M. D., Solomon, A., & Cox, C. J. (2025). Surface energy
    balance responses to radiative forcing in the central Arctic from MOSAiC and
    models. JGR Atmospheres, 130, e2024JD042578.
    https://doi.org/10.1029/2024JD042578

Data source and credentials
===========================
Three CDS datasets, selected with ``--frequency``, all carrying the same
45-variable catalogue on the same 0.25 deg native grid:

    hourly    reanalysis-era5-single-levels
    daily     derived-era5-single-levels-daily-statistics
    monthly   reanalysis-era5-single-levels-monthly-means

Requires the ``cdsapi`` package and a ``~/.cdsapirc`` holding::

    url: https://cds.climate.copernicus.eu/api
    key: <your personal access token>

The current CDS endpoint has no ``/v2`` suffix and the key is the bare token,
not the old ``UID:KEY`` pair. Each dataset also carries its own licence, which
must be accepted once on the CDS website before any request to it will succeed.


OPTIONS
=======

Where the files go
------------------
``--storage {local,external}``
    Which preconfigured root to write under. ``local`` (the default) is the
    ``data/`` directory beside this script; ``external`` is ``EXTERNAL_ROOT``,
    set near the top of this file. Both are edited there, not on the command
    line. If the external volume is not mounted the run aborts rather than
    quietly creating the directory tree on the boot drive and filling it.

``--out-dir PATH``
    Write somewhere else entirely, ignoring ``--storage``. No mount check is
    performed on an explicit path.

Files land in ``<root>/<region>/`` for hourly data and
``<root>/<region>_<frequency>/`` for daily and monthly, so that resolutions are
never mixed in one folder. Mixing them would let the analysis scripts silently
concatenate incompatible time axes.

Which part of the Arctic
------------------------
``--region NAME``
    A named box from ``era5_seb_variables.REGIONS``. Currently ``arctic_circle``
    (north of 66.5N), ``arctic_70n``, ``central_arctic`` (north of 80N),
    ``barrow`` (70-80N, 165-150W), ``beaufort_chukchi``, and ``custom``.
    ``custom`` reads ``CUSTOM_REGION`` in ``era5_seb_variables.py``, which is
    meant to be edited for a domain you will reuse.

``--area NORTH WEST SOUTH EAST``
    An ad hoc box in degrees, overriding ``--region``. Note the CDS ordering:
    North, West, South, East, with longitude on [-180, 180]. Use this for a
    one-off domain rather than editing ``CUSTOM_REGION``. Output is labelled
    ``custom_area``.

``--grid DEG``
    Regrid to a coarser resolution. Omit for the native 0.25 deg. This is the
    single strongest lever on download size: cost scales as 1/DEG^2, so 0.5 deg
    is ~4x smaller and 1.0 deg ~16x smaller. For a pan-Arctic multi-decade
    hourly pull it is often the difference between feasible and not.

Which period
------------
``--start YYYY-MM-DD`` / ``--end YYYY-MM-DD``
    Both inclusive, so ``--end 2026-01-07`` keeps all 24 hours of 7 January.
    ERA5 runs from 1940 to roughly five days behind real time. Data within about
    three months of now is preliminary ERA5T rather than final ERA5; check the
    ``expver`` coordinate in the output if that matters.

    At ``--frequency monthly`` only the year and month of these dates are used;
    the day is ignored, since the product has no sub-monthly resolution.

Which variables
---------------
``--var-set {core,recommended,extended}``
    ``core`` (17)
        The six SEB flux terms, four cloud-cover fields, five column
        hydrometeor fields, total precipitation, and cloud base height. Note
        that liquid and ice water path live here, under ERA5's names
        ``total_column_cloud_liquid_water`` and
        ``total_column_cloud_ice_water``.
    ``recommended`` (35, the default)
        ``core`` plus what Equation (1) actually needs to close and what
        restricts the analysis to Arctic Ocean sea ice: skin temperature, sea
        ice cover, the four sea-ice temperature layers, forecast albedo, the
        four clear-sky fluxes, column water vapour, SST, and near-surface state.
    ``extended`` (44)
        ``recommended`` plus boundary layer height, friction velocity,
        precipitation rates, land snow properties, and TOA fluxes.

    The full registry, with units and the Equation (1) role of each variable,
    is ``era5_seb_variables.py``.

Temporal resolution
-------------------
``--frequency {hourly,daily,monthly}``
    ``hourly``  (default) the full reanalysis, 24 samples per day.
    ``daily``   the derived daily-statistics product. Computed on demand by the
                CDS rather than served from archive, and in testing markedly
                slower per request than either of the other two.
    ``monthly`` the monthly-means product, roughly 730x smaller than hourly.

    Verified equivalence: for a complete month, the monthly product matches the
    monthly average of the hourly files to within 0.003% on skin temperature and
    sea ice and 0.4 W m-2 per cell on the turbulent fluxes.

    The catch is masking. A monthly mean cannot be filtered to ice-free
    conditions after the fact: a cell whose monthly-mean sea ice fraction is 0.5
    was ice-covered for about half the month, and its mean flux already blends
    open-water and ice-covered hours. Masking has to happen before averaging.
    Daily means do not have this problem in practice, because sea ice
    concentration evolves over days rather than hours.

``--daily-statistic {daily_mean,daily_sum,daily_maximum,daily_minimum}``
    Which statistic the daily product reduces each day to. Only meaningful with
    ``--frequency daily``. ``daily_mean`` is the default and is what matches an
    average of the hourly data. Always computed from all 24 hourly steps in UTC.

``--monthly-product {monthly_averaged_reanalysis,monthly_averaged_reanalysis_by_hour_of_day}``
    Only meaningful with ``--frequency monthly``. The default returns one value
    per month. ``by_hour_of_day`` instead returns a mean diurnal cycle, 24
    values per month, which is 24x larger and useful only if the diurnal cycle
    is itself the subject.

How the run is organised
------------------------
``--chunk {day,month,year}``
    How much data goes into each netCDF file and therefore each CDS request.
    Defaults to ``day`` for hourly, ``month`` for daily, and ``year`` for
    monthly. Smaller chunks mean more requests but finer-grained resume and
    less lost work when one fails; larger chunks mean fewer queue waits.
    ``year`` is only a valid request shape for the monthly product and is
    silently downgraded to ``month`` for the others.

``--overwrite``
    Re-download chunks whose output file already exists. Without it, existing
    files are skipped, which is what makes an interrupted run resumable.

``--dry-run``
    Print the plan, the chunk list, the full variable table, and an uncompressed
    size estimate, without contacting the CDS. Always worth running first on a
    large pull.


SIZING
======
Measured from real downloads by this script, at 0.25 deg with the
``recommended`` 35-variable set:

    hourly    1.11 bytes per cell-hour-variable   (3.60x compression)
    daily     1.77 bytes per cell-day-variable    (2.26x compression)

Multiply by grid cells, samples, and variable count. For 2000-2025 (26 years):

    barrow (2,501 cells)          hourly  20 GB      daily  1.3 GB
    arctic_circle (136,800)       hourly  1.07 TB    daily   73 GB
    arctic_circle at --grid 0.5   hourly  277 GB     daily   19 GB

``--dry-run`` reports the uncompressed upper bound for whatever you ask for;
divide by roughly 3.6 (hourly) or 2.3 (daily) for the on-disk figure.


BEHAVIOUR WORTH KNOWING
=======================
Resume
    Resume works in DATE space, not filename space. Before planning, the run
    scans the output directory, decodes which days each existing file covers from
    its name, and drops those days. A range already downloaded as 275 one-day
    files is therefore recognised by a later ``--chunk-days 15`` run, which a
    filename-equality check could never do. Holes left by earlier failures are
    picked up automatically as their own small chunks.

    Chunks are always contiguous day runs, so a ``DD-DD`` filename is an honest
    description of its contents. That matters for correctness, not neatness: a
    gapped chunk named ``05-23`` holding only the 5th and 23rd would make the
    next resume skip the 6th through 22nd.

    ``--overwrite`` bypasses the scan. The raw CDS payload lands on a
    ``.raw.part`` scratch path and the merged netCDF is moved into place only
    once complete, so an interrupted run never leaves a partial file that a later
    resume mistakes for a finished one.

Interrupting a run
    Ctrl-C stops the local process but does NOT cancel the request already
    submitted to the CDS. It keeps running server-side and holds one of your few
    concurrent queue slots; several abandoned jobs will throttle everything that
    follows. Check and clear them at https://cds.climate.copernicus.eu/requests

Retries
    Transfer failures back off and retry up to four times (30 s, 60 s, 120 s).
    Failures that will not fix themselves are not retried at all: licence and
    other 4xx-style rejections stop immediately with a pointer to the CDS
    licence page, and a failure during merging keeps the payload as
    ``.raw_unmerged`` for inspection rather than burning another queue wait.

Consolidation
    The CDS returns a zip of several netCDF files split by GRIB ``stepType``
    whenever a request mixes instantaneous, time-mean, and accumulated fields,
    which ours always does. Those streams are merged into one file per chunk and
    the CDS ``avg_*`` names are renamed to canonical ERA5 short names. See
    README.md for the full mapping and the monthly timestamp quirk.

Manifest
    Each run writes ``manifest_<start>_<end>.json`` recording the dataset,
    frequency, region, area, period, full variable list with units and roles,
    grid, chunking, and the per-file status.


EXAMPLES
========
Plan a large pull without touching the CDS::

    python download_era5_seb.py --dry-run --region arctic_circle \\
        --grid 0.5 --start 2000-01-01 --end 2025-12-31

Smoke test one day to the local disk before committing::

    python download_era5_seb.py --region barrow \\
        --start 2026-01-01 --end 2026-01-01

26 years of hourly data over the Barrow strip, to the external drive (~20 GB)::

    python download_era5_seb.py --storage external --region barrow \\
        --start 2000-01-01 --end 2025-12-31

Pan-Arctic hourly at half resolution, which fits where native does not::

    python download_era5_seb.py --storage external --region arctic_circle \\
        --grid 0.5 --chunk month --start 2000-01-01 --end 2025-12-31

Monthly means for a long climatology, a few hundred MB::

    python download_era5_seb.py --frequency monthly --region arctic_circle \\
        --start 1980-01-01 --end 2025-12-31

An ad hoc box, given as North West South East::

    python download_era5_seb.py --area 82 -60 72 -10
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from era5_seb_variables import (
    VARIABLE_SETS,
    Region,
    cds_variable_list,
    get_region,
    normalise_names,
    region_names,
)

# ----------------------------------------------------------------------------
# Storage roots -- EDIT THESE TO TASTE
# ----------------------------------------------------------------------------
# External drive. The volume is case-insensitive, so "Scripps" also resolves;
# the on-disk casing is used here to match the sibling SCRIPPS/DOE_ARM tree.
EXTERNAL_ROOT = Path("/Volumes/My Passport/SCRIPPS/ERA5/surface_energy_budget")

# Local machine. Sits beside this script. The repository .gitignore already
# excludes *.nc, so downloads here will not be committed.
LOCAL_ROOT = Path(__file__).resolve().parent / "data"

STORAGE_ROOTS: dict[str, Path] = {"external": EXTERNAL_ROOT, "local": LOCAL_ROOT}

# ----------------------------------------------------------------------------
# Datasets, one per temporal resolution
# ----------------------------------------------------------------------------
# All three carry the same 45-variable catalogue (verified against each dataset's
# live CDS schema), so any --var-set works at any frequency. They differ in which
# request keys they accept, which is why build_request dispatches on frequency.
FREQUENCY_DATASETS: dict[str, str] = {
    "hourly": "reanalysis-era5-single-levels",
    "daily": "derived-era5-single-levels-daily-statistics",
    "monthly": "reanalysis-era5-single-levels-monthly-means",
}

# The daily product is a DERIVED dataset: it has no data_format/download_format
# keys and always returns netCDF. Sending those keys makes the request invalid.
FORMAT_AWARE_FREQUENCIES = {"hourly", "monthly"}

DEFAULT_START = date(2026, 1, 1)
DEFAULT_END = date(2026, 1, 7)  # inclusive -- first week of January 2026
DEFAULT_REGION = "arctic_circle"
DEFAULT_VAR_SET = "recommended"
DEFAULT_FREQUENCY = "hourly"
DEFAULT_DAILY_STATISTIC = "daily_mean"
DEFAULT_MONTHLY_PRODUCT = "monthly_averaged_reanalysis"

ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
NATIVE_GRID_DEG = 0.25

# Approximate size ratio relative to hourly, for the size estimate. Daily keeps
# one value per day instead of 24; monthly keeps one per month.
FREQUENCY_SAMPLE_DIVISOR = {"hourly": 1.0, "daily": 24.0, "monthly": 730.0}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def daterange(start_date: date, end_date: date) -> list[date]:
    """Every calendar day from ``start_date`` to ``end_date``, inclusive."""
    if end_date < start_date:
        raise ValueError(f"end date {end_date} precedes start date {start_date}")
    n_days = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(n_days)]


# Empirically probed per-request ceilings, expressed in "hourly source fields"
# = variables x days x 24. The CDS costs a request by how much hourly ERA5 it
# must touch, NOT by the size of the output, which is why the daily product --
# whose output is 24x smaller -- has the TIGHTER ceiling: it still has to read
# all 24 hourly steps to form each daily mean.
#
#   hourly   12,648 accepted   25,296 rejected (403)
#   daily     8,160 accepted   12,240 rejected (403)
#
# These are the largest requests VERIFIED to be accepted. The true threshold was
# not bisected, so it sits somewhere above each of these and below the matching
# rejection. Anything at or under these values is known to work; above them is
# untested rather than known-bad.
SAFE_REQUEST_FIELDS = {"hourly": 12_648, "daily": 8_160, "monthly": None}


# Filename shapes this script writes, mapped back to the days they cover. Resume
# has to work in DATE space rather than filename space: a run chunked one way
# must recognise data already on disk that was chunked another way.
_FILE_DAY_PATTERNS = (
    # ..._20000523.nc  -- a single day (--chunk day)
    (re.compile(r"_(\d{4})(\d{2})(\d{2})$"), "single"),
    # ..._200005_01-15.nc  -- a contiguous day range inside one month
    (re.compile(r"_(\d{4})(\d{2})_(\d{2})-(\d{2})$"), "range"),
    # ..._2000_m01-12.nc  -- whole months (monthly means)
    (re.compile(r"_(\d{4})_m(\d{2})-(\d{2})$"), "months"),
)


def days_covered_by_file(path: Path) -> set[date]:
    """Days a completed output file covers, inferred from its name.

    Sound because the downloader only ever renames a file into place once it is
    complete, and only ever emits contiguous day ranges -- so the name is an
    accurate description of the contents, not a guess.
    """
    stem = path.stem
    for pattern, kind in _FILE_DAY_PATTERNS:
        m = pattern.search(stem)
        if not m:
            continue
        if kind == "single":
            y, mo, d = (int(g) for g in m.groups())
            return {date(y, mo, d)}
        if kind == "range":
            y, mo, d0, d1 = (int(g) for g in m.groups())
            return {date(y, mo, d) for d in range(d0, d1 + 1)}
        y, m0, m1 = (int(g) for g in m.groups())
        out: set[date] = set()
        for mo in range(m0, m1 + 1):
            out.update(
                date(y, mo, d) for d in range(1, calendar.monthrange(y, mo)[1] + 1)
            )
        return out
    return set()


def scan_covered_days(run_dir: Path) -> set[date]:
    """Every day already present in ``run_dir``, across all chunk shapes."""
    if not run_dir.is_dir():
        return set()
    covered: set[date] = set()
    for path in run_dir.glob("*.nc"):
        covered |= days_covered_by_file(path)
    return covered


def contiguous_runs(days: list[date]) -> list[list[date]]:
    """Split a sorted day list into maximal runs of consecutive days.

    Chunks must stay contiguous so that a ``DD-DD`` filename is always an honest
    description of its contents. A gapped chunk named 05-23 that actually held
    only the 5th and 23rd would make a later resume skip the 6th through 22nd.
    """
    runs: list[list[date]] = []
    current: list[date] = []
    for d in sorted(days):
        if current and (d - current[-1]).days == 1:
            current.append(d)
        else:
            if current:
                runs.append(current)
            current = [d]
    if current:
        runs.append(current)
    return runs


def split_evenly(items: list, max_size: int) -> list[list]:
    """Split ``items`` into the fewest near-equal groups of at most ``max_size``.

    Near-equal rather than greedy so a 31-day month at max 15 becomes 11/11/9
    instead of 15/15/1, keeping every request comfortably under the ceiling.
    """
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    n_groups = max(1, -(-len(items) // max_size))  # ceil division
    out, start = [], 0
    for i in range(n_groups):
        size = len(items) // n_groups + (1 if i < len(items) % n_groups else 0)
        out.append(items[start : start + size])
        start += size
    return out


def months_between(start_date: date, end_date: date) -> list[tuple[int, int]]:
    """Every ``(year, month)`` touched by the range, inclusive at both ends."""
    if end_date < start_date:
        raise ValueError(f"end date {end_date} precedes start date {start_date}")
    out: list[tuple[int, int]] = []
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def chunk_by_month(days: list[date]) -> list[tuple[int, int, list[date]]]:
    """Group a day list into ``(year, month, days_in_that_month)`` tuples."""
    grouped: dict[tuple[int, int], list[date]] = {}
    for d in days:
        grouped.setdefault((d.year, d.month), []).append(d)
    return [(y, m, ds) for (y, m), ds in sorted(grouped.items())]


def estimate_size_gb(
    region: Region,
    n_hours: int,
    n_vars: int,
    grid_deg: float = NATIVE_GRID_DEG,
    frequency: str = "hourly",
) -> float:
    """Uncompressed upper bound on the download size, in GB.

    Assumes float32 storage on a regular lat/lon grid. Real netCDF output is
    typically 2-4x smaller because CDS applies compression, so treat this as a
    ceiling rather than an expectation.

    ``n_hours`` is the span in hours; the frequency divisor converts that into
    the number of stored time samples.
    """
    n_lat = round((region.north_deg - region.south_deg) / grid_deg) + 1
    spans_globe = region.west_deg <= -180.0 and region.east_deg >= 180.0
    if spans_globe:
        n_lon = round(360.0 / grid_deg)  # wraps, so no +1
    else:
        n_lon = round((region.east_deg - region.west_deg) / grid_deg) + 1
    n_samples = max(1.0, n_hours / FREQUENCY_SAMPLE_DIVISOR.get(frequency, 1.0))
    n_bytes = n_lat * n_lon * n_samples * n_vars * 4
    return n_bytes / 1024**3


def check_volume_mounted(root: Path) -> None:
    """Raise if ``root`` sits on a /Volumes mount that is not currently mounted.

    Guards against silently creating a directory tree on the boot drive at the
    external path when the drive is unplugged, which would then fill it.
    """
    parts = root.parts
    if len(parts) < 3 or parts[1] != "Volumes":
        return  # not an external volume path; nothing to check
    volume = Path(parts[0], parts[1], parts[2])
    if not volume.is_dir():
        raise FileNotFoundError(
            f"External volume {volume} is not mounted. Connect the drive, rerun "
            f"with --storage local, or pass --out-dir explicitly."
        )


def describe_chunk(
    year: int | None,
    month: int | None,
    days: list[date] | None,
    months: list[int] | None,
) -> str:
    """One-line human description of a chunk, for logs and dry runs."""
    if months is not None:
        return f"{year} months {months[0]:02d}-{months[-1]:02d}"
    if days:
        span = f"{days[0].day:02d}-{days[-1].day:02d}"
        return f"{year}-{month:02d} day{'s' if len(days) > 1 else ''} {span}"
    return f"{year}-{month:02d}"


def resolve_output_root(storage: str, out_dir: str | None) -> Path:
    """Pick the output directory. Does not create it."""
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve()
    return STORAGE_ROOTS[storage]


def verify_credentials() -> bool:
    """Check that ``~/.cdsapirc`` exists and points at the current CDS endpoint."""
    rc_path = Path.home() / ".cdsapirc"
    if not rc_path.exists():
        print(f"  [x] No CDS credentials at {rc_path}")
        print("      Register at https://cds.climate.copernicus.eu/, then create")
        print("      ~/.cdsapirc containing:")
        print("        url: https://cds.climate.copernicus.eu/api")
        print("        key: <your personal access token>")
        return False

    text = rc_path.read_text()
    print(f"  [ok] Found CDS credentials at {rc_path}")
    if "/api/v2" in text:
        print("  [!] Warning: that file still uses the retired /api/v2 endpoint.")
        print("      Change the url line to https://cds.climate.copernicus.eu/api")
    return True


# ----------------------------------------------------------------------------
# Request construction and download
# ----------------------------------------------------------------------------
def build_request(
    frequency: str,
    variables: list[str],
    area: list[float],
    year: int | None = None,
    month: int | None = None,
    days: list[date] | None = None,
    months: list[int] | None = None,
    grid_deg: float | None = None,
    daily_statistic: str = DEFAULT_DAILY_STATISTIC,
    monthly_product: str = DEFAULT_MONTHLY_PRODUCT,
) -> dict:
    """Assemble one CDS request for the given frequency.

    The three datasets take different keys:

    hourly   year, month, day, time (24 h), product_type, data/download_format
    daily    year, month, day, daily_statistic, frequency, time_zone (no format keys)
    monthly  year, month, time, product_type (no day)
    """
    if frequency == "hourly":
        request = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d.day:02d}" for d in days or []],
            "time": ALL_HOURS,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        }

    elif frequency == "daily":
        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": str(year),
            "month": [f"{month:02d}"],
            "day": [f"{d.day:02d}" for d in days or []],
            "daily_statistic": daily_statistic,
            "time_zone": "utc+00:00",
            # Which sub-daily steps feed the statistic. 1_hourly uses all 24,
            # matching what an hourly download would average to.
            "frequency": "1_hourly",
            "area": area,
        }

    elif frequency == "monthly":
        # "time" is still required. For a plain monthly mean it is a single
        # placeholder hour; only the by_hour_of_day product uses all 24 to
        # return a mean diurnal cycle for each month.
        by_hour = monthly_product.endswith("by_hour_of_day")
        request = {
            "product_type": monthly_product,
            "variable": variables,
            "year": [str(year)],
            "month": [f"{m:02d}" for m in (months or [])],
            "time": ALL_HOURS if by_hour else ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        }

    else:
        raise ValueError(
            f"Unknown frequency {frequency!r}; choose from {sorted(FREQUENCY_DATASETS)}."
        )

    if grid_deg is not None:
        request["grid"] = [grid_deg, grid_deg]
    return request


def _floor_time_to_month(ds):
    """Snap ``valid_time`` to the first instant of its month.

    The monthly-means product splits into stepType streams that timestamp the
    SAME calendar month at different hours: the time-mean flux streams (avgad,
    avgid) land on 06:00 of the first, while the instantaneous stream (avgua)
    lands on 00:00. Merging with join="exact" fails on that 6-hour offset even
    though both describe the same month, so the hour is discarded first.

    Only applied to monthly data. At hourly or daily resolution a time offset is
    a real difference and must not be flattened.
    """
    if "valid_time" not in ds.coords:
        return ds
    month_start = ds["valid_time"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return ds.assign_coords(valid_time=month_start)


def consolidate_to_netcdf(
    raw_path: Path, final_path: Path, align_to_month: bool = False
) -> None:
    """Turn whatever the CDS actually returned into one tidy netCDF file.

    The CDS splits a request into separate netCDF files by GRIB ``stepType`` and
    returns them zipped whenever a request mixes types, regardless of
    ``download_format: unarchived``. Our variable list always mixes instantaneous
    fields (skin temperature, cloud cover), time-mean fluxes, and accumulations
    (total precipitation), so a zip of three files is the normal result.

    This merges those streams onto their shared (valid_time, latitude, longitude)
    grid and renames the CDS ``avg_*`` fields to canonical ERA5 short names, so
    downstream code sees one file with the names the registry advertises. A plain
    netCDF response is passed through the same normalisation.
    """
    import xarray as xr

    def _prepare(d):
        d = normalise_names(d)
        return _floor_time_to_month(d) if align_to_month else d

    if not zipfile.is_zipfile(raw_path):
        ds = xr.open_dataset(raw_path)
        try:
            out = _prepare(ds).load()
        finally:
            ds.close()
        out.to_netcdf(final_path, encoding=_compression_encoding(out))
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(raw_path) as zf:
            members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not members:
                raise RuntimeError(f"{raw_path.name} contains no .nc members")
            zf.extractall(tmp_dir, members)

        opened = [xr.open_dataset(Path(tmp_dir) / m) for m in members]
        try:
            # join="exact" so a coordinate mismatch between streams fails loudly
            # rather than silently broadcasting into a larger array.
            # compat is pinned rather than left default: the streams carry
            # disjoint data variables but shared coordinates, and "no_conflicts"
            # verifies those coordinates agree instead of letting one win.
            merged = xr.merge(
                [_prepare(d) for d in opened],
                join="exact",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            ).load()
        finally:
            for d in opened:
                d.close()

        merged.attrs["cds_stream_files"] = ", ".join(sorted(members))
        merged.to_netcdf(final_path, encoding=_compression_encoding(merged))


# Substrings that mark a CDS error as permanent rather than transient. Retrying
# these just burns backoff: the request will be rejected identically every time.
# The most common one in practice is the separate licence agreement that the
# derived-* datasets require you to accept once on the CDS website.
# Patterns that mark a CDS error as permanent rather than transient. Retrying
# these just burns backoff: the request will be rejected identically every time.
# The most common one in practice is the separate licence agreement that each
# dataset requires you to accept once on the CDS website.
#
# These are regexes with word boundaries, not bare substrings, deliberately. A
# bare "400" test matches the hex object address in a message like
#   "<urllib3.connection.HTTPSConnection object at 0x1204009d0>: Failed to
#    resolve 'cds.climate.copernicus.eu'"
# which is a DNS outage -- exactly the transient case that must keep retrying.
# HTTP status codes therefore only count when adjacent to a word that makes them
# unambiguously a status code.
PERMANENT_ERROR_PATTERNS = (
    r"licen[cs]e",
    r"\bnot valid\b",
    r"\binvalid\b",
    r"\bforbidden\b",
    r"\bunauthori[sz]ed\b",
    r"\b40[0-4]\s+client error",
    r"\bhttp\s*40[0-4]\b",
    r"\bstatus[ _]?(?:code)?[ =:]+40[0-4]\b",
)


# Signals that a request was ACCEPTED and then failed while running server-side.
# The CDS reports this as a 400 on the job's /results endpoint, which looks like a
# client error but is not one: the request was well-formed enough to be queued and
# executed. These failures are frequently transient (a bad worker, a transient
# backend fault) and a resubmission of the identical request usually succeeds, so
# they must stay retryable. Checked before PERMANENT_ERROR_PATTERNS, whose "400
# client error" rule would otherwise swallow them.
SERVER_SIDE_JOB_FAILURE_PATTERNS = (
    r"the job has failed",
    r"/results\b",
    r"\bjob (?:has )?failed\b",
)


def _is_permanent_error(exc: Exception) -> bool:
    """True when a CDS failure will not be fixed by trying again.

    A malformed or unlicensed request is rejected at submission. A request that
    reached "running" and then failed is a different animal and is treated as
    transient, even though the CDS surfaces it with a 400 status.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(re.search(p, text) for p in SERVER_SIDE_JOB_FAILURE_PATTERNS):
        return False
    return any(re.search(p, text) for p in PERMANENT_ERROR_PATTERNS)


def _compression_encoding(ds, complevel: int = 4) -> dict:
    """zlib encoding for every float field, to keep the merged file compact."""
    return {
        name: {"zlib": True, "complevel": complevel}
        for name, da in ds.data_vars.items()
        if da.dtype.kind == "f"
    }


# One cdsapi client per worker thread. The client keeps a requests.Session and
# per-request polling state, neither of which is documented thread-safe, so
# sharing a single instance across the pool would be asking for trouble.
_thread_local = threading.local()


def _thread_client():
    """The calling thread's cdsapi client, created on first use."""
    import cdsapi

    if not hasattr(_thread_local, "client"):
        _thread_local.client = cdsapi.Client()
    return _thread_local.client


def download_one(
    client,
    dataset: str,
    request: dict,
    output_path: Path,
    max_retries: int = 4,
    align_to_month: bool = False,
) -> tuple[bool, float]:
    """Retrieve one CDS request with exponential backoff, then consolidate it.

    Returns ``(success, size_mb)``. The raw CDS payload lands on a scratch path
    and the merged netCDF is only moved into place once it is complete, so an
    interrupted run never leaves a partial file that a later resume would mistake
    for a finished one.
    """
    raw_path = output_path.with_suffix(".raw.part")
    merged_path = output_path.with_suffix(".merged.part")

    def _cleanup() -> None:
        raw_path.unlink(missing_ok=True)
        merged_path.unlink(missing_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            client.retrieve(dataset, request).download(str(raw_path))
        except KeyboardInterrupt:
            _cleanup()
            raise
        except Exception as exc:  # cdsapi raises bare Exception on HTTP errors
            print(f"      attempt {attempt}/{max_retries} failed: {exc}")
            _cleanup()
            if _is_permanent_error(exc):
                print("      this looks permanent, not transient -- not retrying.")
                # Point at the fix that matches what the CDS actually said,
                # rather than guessing at a licence every time.
                low = str(exc).lower()
                if "too large" in low or "cost limit" in low:
                    print("      the CDS rejected this request as too large. Use a")
                    print("      smaller --chunk, a smaller --var-set, or a coarser")
                    print("      --grid. The derived daily product has a much lower")
                    print("      cost ceiling than the hourly archive.")
                elif "licen" in low:
                    print("      accept this dataset's licence on the CDS website,")
                    print("      then rerun: https://cds.climate.copernicus.eu/")
                else:
                    print("      check the request against the dataset's CDS page.")
                return False, 0.0
            if attempt < max_retries:
                wait_s = 30 * 2 ** (attempt - 1)
                print(f"      retrying in {wait_s} s ...")
                time.sleep(wait_s)
                continue
            print("      giving up on this chunk")
            return False, 0.0

        # The transfer succeeded. A failure from here on is deterministic — the
        # payload is on disk and will not merge differently next time — so it is
        # NOT retried, and the raw file is kept for inspection rather than
        # deleted. Retrying it would just burn another CDS queue wait.
        try:
            consolidate_to_netcdf(raw_path, merged_path, align_to_month=align_to_month)
        except KeyboardInterrupt:
            _cleanup()
            raise
        except Exception as exc:
            kept = output_path.with_suffix(".raw_unmerged")
            raw_path.replace(kept)
            merged_path.unlink(missing_ok=True)
            print(f"      download OK but consolidation failed: {exc}")
            print(f"      raw payload kept at {kept.name} for inspection")
            return False, 0.0

        merged_path.replace(output_path)
        raw_path.unlink(missing_ok=True)
        return True, output_path.stat().st_size / 1024**2

    return False, 0.0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
CLI_EPILOG = """\
examples:
  # plan a large pull without touching the CDS
  download_era5_seb.py --dry-run --region arctic_circle --grid 0.5 \\
      --start 2000-01-01 --end 2025-12-31

  # smoke test one day locally before committing
  download_era5_seb.py --region barrow --start 2026-01-01 --end 2026-01-01

  # 26 years hourly over the Barrow strip, to the external drive (~20 GB)
  download_era5_seb.py --storage external --region barrow \\
      --start 2000-01-01 --end 2025-12-31

  # pan-Arctic hourly at half resolution (~277 GB; native would be 1.07 TB)
  download_era5_seb.py --storage external --region arctic_circle --grid 0.5 \\
      --chunk month --start 2000-01-01 --end 2025-12-31

  # monthly means for a long climatology (a few hundred MB)
  download_era5_seb.py --frequency monthly --region arctic_circle \\
      --start 1980-01-01 --end 2025-12-31

sizing (0.25 deg, 34 variables, measured from real downloads):
  hourly  1.11 bytes per cell-hour-variable   daily  1.77 bytes per cell-day-variable
  26 years: barrow 20 GB hourly / 1.3 GB daily
            arctic_circle 1.07 TB hourly / 73 GB daily
            arctic_circle --grid 0.5  277 GB hourly / 19 GB daily

See the module docstring in this file, or README.md, for the full reference.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_era5_seb.py",
        description=(
            "Download ERA5 single-level data for an Arctic surface energy budget "
            "analysis (Sledd et al. 2025, Eq. 1). Selects hourly, daily, or "
            "monthly data from the corresponding CDS dataset, splits the request "
            "into resumable chunks, merges the multi-stream CDS response into one "
            "netCDF per chunk, and writes a run manifest."
        ),
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Where the files go --------------------------------------------------
    g_out = parser.add_argument_group("output location")
    g_out.add_argument(
        "--storage",
        choices=sorted(STORAGE_ROOTS),
        default="local",
        help=(
            "Which preconfigured root to write under. 'local' is the data/ "
            "directory beside this script; 'external' is EXTERNAL_ROOT, set near "
            "the top of this file. If the external volume is not mounted the run "
            "aborts rather than filling the boot drive. (default: local)"
        ),
    )
    g_out.add_argument(
        "--out-dir",
        default=None,
        metavar="PATH",
        help=(
            "Write somewhere else entirely, ignoring --storage. No mount check is "
            "performed on an explicit path."
        ),
    )

    # --- Which part of the Arctic -------------------------------------------
    g_space = parser.add_argument_group("spatial domain")
    g_space.add_argument(
        "--region",
        choices=region_names(),
        default=DEFAULT_REGION,
        help=(
            "Named box from era5_seb_variables.REGIONS. 'custom' reads "
            "CUSTOM_REGION in that file, for a domain you will reuse. "
            f"(default: {DEFAULT_REGION})"
        ),
    )
    g_space.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=None,
        help=(
            "Ad hoc box in degrees, overriding --region. Note the CDS ordering: "
            "North West South East, longitude on [-180, 180]. Output is labelled "
            "'custom_area'."
        ),
    )
    g_space.add_argument(
        "--grid",
        type=float,
        default=None,
        metavar="DEG",
        help=(
            "Regrid to a coarser resolution. Omit for native 0.25 deg. The "
            "strongest lever on download size: cost scales as 1/DEG^2, so 0.5 is "
            "~4x smaller and 1.0 is ~16x smaller."
        ),
    )

    # --- Which period --------------------------------------------------------
    g_time = parser.add_argument_group("time period")
    g_time.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_START,
        metavar="YYYY-MM-DD",
        help=f"First day, inclusive. ERA5 begins in 1940. (default: {DEFAULT_START})",
    )
    g_time.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_END,
        metavar="YYYY-MM-DD",
        help=(
            "Last day, inclusive, so --end 2026-01-07 keeps all 24 hours of 7 Jan. "
            "ERA5 runs to about five days behind real time; data within ~3 months "
            "of now is preliminary ERA5T (check the expver coordinate). At "
            f"--frequency monthly only the year and month are used. "
            f"(default: {DEFAULT_END})"
        ),
    )

    # --- Which variables -----------------------------------------------------
    g_vars = parser.add_argument_group("variables")
    g_vars.add_argument(
        "--var-set",
        choices=sorted(VARIABLE_SETS),
        default=DEFAULT_VAR_SET,
        help=(
            f"core ({len(VARIABLE_SETS['core'])}) = the SEB flux terms, cloud "
            "cover, column hydrometeors incl. liquid and ice water path, "
            "precipitation, cloud base height. "
            f"recommended ({len(VARIABLE_SETS['recommended'])}) = core plus what "
            "closes Eq. (1) and masks to sea ice: skin temperature, sea ice cover, "
            "ice temperature layers, albedo, clear-sky fluxes, column water "
            "vapour, near-surface state. "
            f"extended ({len(VARIABLE_SETS['extended'])}) = recommended plus "
            "boundary layer height, friction velocity, precip rates, land snow, "
            f"SST, TOA fluxes. Full registry with units in era5_seb_variables.py. "
            f"(default: {DEFAULT_VAR_SET})"
        ),
    )

    # --- Temporal resolution -------------------------------------------------
    g_freq = parser.add_argument_group("temporal resolution")
    g_freq.add_argument(
        "--frequency",
        choices=sorted(FREQUENCY_DATASETS),
        default=DEFAULT_FREQUENCY,
        help=(
            "'hourly' is the full reanalysis. 'daily' is the derived "
            "daily-statistics product, computed on demand by the CDS and slower "
            "per request. 'monthly' is ~730x smaller than hourly and numerically "
            "matches an average of the hourly data, BUT a monthly mean cannot be "
            "masked to ice-free conditions after the fact -- masking must precede "
            f"averaging. (default: {DEFAULT_FREQUENCY})"
        ),
    )
    g_freq.add_argument(
        "--daily-statistic",
        choices=("daily_mean", "daily_sum", "daily_maximum", "daily_minimum"),
        default=DEFAULT_DAILY_STATISTIC,
        help=(
            "Which statistic each day is reduced to. Only used with --frequency "
            "daily. Always computed from all 24 hourly steps in UTC; daily_mean is "
            f"what matches an average of the hourly data. "
            f"(default: {DEFAULT_DAILY_STATISTIC})"
        ),
    )
    g_freq.add_argument(
        "--monthly-product",
        choices=(
            "monthly_averaged_reanalysis",
            "monthly_averaged_reanalysis_by_hour_of_day",
        ),
        default=DEFAULT_MONTHLY_PRODUCT,
        help=(
            "Only used with --frequency monthly. The default returns one value per "
            "month; 'by_hour_of_day' returns a mean diurnal cycle (24 values per "
            "month), 24x larger and useful only if the diurnal cycle is the "
            f"subject. (default: {DEFAULT_MONTHLY_PRODUCT})"
        ),
    )

    # --- How the run is organised -------------------------------------------
    g_run = parser.add_argument_group("run behaviour")
    g_run.add_argument(
        "--chunk",
        choices=("day", "month", "year"),
        default=None,
        help=(
            "How much data goes in each file, and therefore each CDS request. "
            "Smaller chunks mean more requests but finer-grained resume; larger "
            "chunks mean fewer queue waits. 'year' is only valid for monthly and "
            "is downgraded to 'month' otherwise. "
            "(default: day for hourly, month for daily, year for monthly)"
        ),
    )
    g_run.add_argument(
        "--chunk-days",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Put at most N days in each request, overriding --chunk. Chunks never "
            "span a month boundary, and each month is split into near-equal "
            "groups. Use this to sit just under the CDS per-request ceiling: the "
            "cost is variables x days x 24, capped near 12,000 for hourly and "
            "8,000 for daily. With 34 variables that is about 15 days (hourly) or "
            "10 days (daily); with the 17-variable core set a full month fits."
        ),
    )
    g_run.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of CDS requests to keep in flight at once (default 1). "
            "Wall time is dominated by queue latency, not bandwidth -- a single "
            "request can sit 'accepted' for an hour -- so N=3 or 4 can be several "
            "times faster. Do not go high: the CDS throttles users with many "
            "queued jobs, which is the failure mode that made a serial run "
            "degrade from 1.8 to 34 minutes per request."
        ),
    )
    g_run.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-download chunks whose output file already exists. Without this, "
            "existing files are skipped, which is what makes a run resumable."
        ),
    )
    g_run.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the plan, chunk list, variable table, and uncompressed size "
            "estimate without contacting the CDS. Divide the estimate by ~3.6 "
            "(hourly) or ~2.3 (daily) for the on-disk figure."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Resolve the spatial domain -----------------------------------------
    if args.area is not None:
        north_deg, west_deg, south_deg, east_deg = args.area
        if north_deg <= south_deg:
            raise SystemExit(
                f"--area NORTH ({north_deg}) must exceed SOUTH ({south_deg})."
            )
        region = Region(north_deg, west_deg, south_deg, east_deg, "explicit --area")
        region_label = "custom_area"
    else:
        region = get_region(args.region)
        region_label = args.region

    variables = cds_variable_list(args.var_set)
    try:
        days = daterange(args.start, args.end)
    except ValueError as exc:
        raise SystemExit(f"{exc}. Check --start and --end.") from None
    n_hours = 24 * len(days)

    dataset = FREQUENCY_DATASETS[args.frequency]
    chunk = args.chunk or {"hourly": "day", "daily": "month", "monthly": "year"}[
        args.frequency
    ]
    if args.frequency == "monthly" and chunk != "year":
        print(
            f"  Note: --chunk {chunk} is ignored for monthly data; chunking by year.",
            file=sys.stderr,
        )
        chunk = "year"
    elif args.frequency != "monthly" and chunk == "year":
        # Year chunks carry a month list and no day list, which is only a valid
        # request shape for the monthly-means dataset. The hourly and daily
        # datasets both require explicit days, so this would build an empty
        # "day" field and be rejected by the CDS.
        print(
            f"  Note: --chunk year is only valid for --frequency monthly; "
            f"using month chunks for {args.frequency} data.",
            file=sys.stderr,
        )
        chunk = "month"

    # --- Plan ----------------------------------------------------------------
    print("=" * 72)
    print("ERA5 surface energy budget download")
    print("=" * 72)
    print(f"  Frequency    : {args.frequency}")
    print(f"  Dataset      : {dataset}")
    if args.frequency == "daily":
        print(f"  Statistic    : {args.daily_statistic} (from 1-hourly steps, UTC)")
    elif args.frequency == "monthly":
        print(f"  Product      : {args.monthly_product}")
    print(f"  Region       : {region_label} -- {region.description}")
    print(f"  Area [N W S E]: {region.as_area()}")
    period_unit = {"hourly": "hourly", "daily": "daily", "monthly": "monthly"}[
        args.frequency
    ]
    print(f"  Period       : {args.start} to {args.end} ({len(days)} days, {period_unit})")
    print(f"  Variable set : {args.var_set} ({len(variables)} variables)")
    print(f"  Grid         : {args.grid or NATIVE_GRID_DEG} deg")
    if args.chunk_days is not None and args.frequency != "monthly":
        chunk_label = f"up to {args.chunk_days} days per file"
    else:
        chunk_label = f"one file per {chunk}"
    print(f"  Chunking     : {chunk_label}")
    est_gb = estimate_size_gb(
        region, n_hours, len(variables), args.grid or NATIVE_GRID_DEG, args.frequency
    )
    print(f"  Size estimate: ~{est_gb:.2f} GB uncompressed (netCDF will be smaller)")

    output_root = resolve_output_root(args.storage, args.out_dir)
    # Keep frequencies in separate subdirectories: mixing monthly means and
    # hourly files in one folder would let the analysis scripts silently
    # concatenate incompatible time axes.
    subdir = region_label if args.frequency == "hourly" else f"{region_label}_{args.frequency}"
    run_dir = output_root / subdir
    print(f"  Output       : {run_dir}")
    print("-" * 72)

    # --- Build the chunk list ------------------------------------------------
    # Each entry is (filename, year, month, days, months).
    # Hourly filenames keep their original form so that files downloaded before
    # the frequency option existed still match and are skipped on resume.
    prefix = (
        f"era5_seb_{region_label}"
        if args.frequency == "hourly"
        else f"era5_seb_{args.frequency}_{region_label}"
    )
    if args.chunk_days is not None:
        if args.frequency == "monthly":
            print("  Note: --chunk-days does not apply to monthly data; ignoring.",
                  file=sys.stderr)
            args.chunk_days = None
        elif args.chunk_days < 1:
            raise SystemExit("--chunk-days must be at least 1.")

    # --- Skip days already on disk, whatever chunk shape produced them -------
    # A filename-only check cannot do this: 15-day chunks and per-day files have
    # different names for the same dates, so a resume would re-download data it
    # already has.
    if not args.overwrite:
        covered = scan_covered_days(run_dir)
        if covered:
            remaining = [d for d in days if d not in covered]
            n_have = len(days) - len(remaining)
            if n_have:
                print(f"  Already on disk: {n_have:,} of {len(days):,} days")
            if not remaining:
                print("\nEvery requested day is already present. Nothing to do.")
                print("Use --overwrite to re-download.")
                return 0
            days = remaining

    if args.chunk_days is not None:
        chunks = []
        for y, m, month_days in chunk_by_month(days):
            # Contiguous runs first, so every emitted DD-DD range is truthful
            # even when resume leaves holes mid-month.
            for run in contiguous_runs(month_days):
                for group in split_evenly(run, args.chunk_days):
                    chunks.append((
                        f"{prefix}_{y:04d}{m:02d}"
                        f"_{group[0].day:02d}-{group[-1].day:02d}.nc",
                        y, m, group, None,
                    ))
    elif chunk == "day":
        chunks = [(f"{prefix}_{d:%Y%m%d}.nc", d.year, d.month, [d], None) for d in days]
    elif chunk == "month":
        chunks = [
            (
                f"{prefix}_{y:04d}{m:02d}_{run[0].day:02d}-{run[-1].day:02d}.nc",
                y,
                m,
                run,
                None,
            )
            for y, m, month_days in chunk_by_month(days)
            # Contiguous runs, for the same filename-honesty reason as above.
            for run in contiguous_runs(month_days)
        ]
    else:  # year -- monthly means, all months of a year in one request
        by_year: dict[int, list[int]] = {}
        for y, m in months_between(args.start, args.end):
            by_year.setdefault(y, []).append(m)
        chunks = [
            (
                f"{prefix}_{y:04d}_m{ms[0]:02d}-{ms[-1]:02d}.nc",
                y,
                None,
                None,
                ms,
            )
            for y, ms in sorted(by_year.items())
        ]

    # --- Preflight: will the CDS accept a request this large? ----------------
    # Cheaper to catch here than to discover it after a queue wait.
    safe_fields = SAFE_REQUEST_FIELDS.get(args.frequency)
    if safe_fields is not None and chunks:
        max_days = max(len(c[3] or []) for c in chunks)
        fields = len(variables) * max_days * 24
        print(f"  Request cost : {fields:,} hourly source fields per request "
              f"(largest chunk = {max_days} day{'s' if max_days > 1 else ''})")
        if fields > safe_fields:
            n_ok = max(1, safe_fields // (len(variables) * 24))
            # stdout, not stderr, so it stays in order with the rest of the plan.
            print(
                f"\n  !! This exceeds the largest {args.frequency} request verified to\n"
                f"  !! be accepted ({safe_fields:,} fields). The CDS may reject it with\n"
                f"  !! a 403. Try --chunk-days {n_ok}, or a smaller --var-set."
            )
    print(f"  Requests     : {len(chunks):,}")

    if args.dry_run:
        print("Dry run -- the following chunks would be requested:")
        for filename, year, month, chunk_days, chunk_months in chunks:
            print(f"  {filename}  ({describe_chunk(year, month, chunk_days, chunk_months)})")
        print()
        print("Variables:")
        for v in VARIABLE_SETS[args.var_set]:
            print(f"  {v.short_name:<11} {v.units:<16} {v.cds_name}  [{v.role}]")
        print()
        try:
            check_volume_mounted(run_dir)
        except FileNotFoundError as exc:
            print(f"Note: {exc}")
        print("No data was downloaded. Drop --dry-run to execute.")
        return 0

    # --- Credentials and client ---------------------------------------------
    check_volume_mounted(run_dir)
    print("Checking CDS credentials ...")
    if not verify_credentials():
        return 1
    try:
        import cdsapi
    except ImportError:
        print("  [x] cdsapi is not installed. Run: pip install cdsapi")
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    print()

    # --- Download -------------------------------------------------------------
    manifest_entries: list[dict] = []
    n_done = n_skipped = n_failed = 0
    total_mb = 0.0
    t_start = time.monotonic()
    print_lock = threading.Lock()

    # Skip existing files up front so the pool only carries real work.
    pending = []
    for filename, year, month, chunk_days, chunk_months in chunks:
        output_path = run_dir / filename
        if output_path.exists() and not args.overwrite:
            size_mb = output_path.stat().st_size / 1024**2
            n_skipped += 1
            manifest_entries.append(
                {"file": filename, "status": "skipped_existing", "size_mb": round(size_mb, 2)}
            )
            continue
        pending.append((filename, year, month, chunk_days, chunk_months))

    if n_skipped:
        print(f"  {n_skipped} chunk(s) already on disk -- skipping")
    if not pending:
        print("\nNothing left to download.")

    def _run_chunk(task):
        filename, year, month, chunk_days, chunk_months = task
        request = build_request(
            frequency=args.frequency,
            variables=variables,
            area=region.as_area(),
            year=year,
            month=month,
            days=chunk_days,
            months=chunk_months,
            grid_deg=args.grid,
            daily_statistic=args.daily_statistic,
            monthly_product=args.monthly_product,
        )
        ok, size_mb = download_one(
            _thread_client(),
            dataset,
            request,
            run_dir / filename,
            align_to_month=(args.frequency == "monthly"),
        )
        return filename, describe_chunk(year, month, chunk_days, chunk_months), ok, size_mb

    n_total = len(pending)
    completed = 0
    if args.jobs > 1:
        print(f"  running {args.jobs} requests concurrently\n")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_chunk, t): t for t in pending}
        try:
            for fut in as_completed(futures):
                filename, span, ok, size_mb = fut.result()
                with print_lock:
                    completed += 1
                    tag = f"done ({size_mb:.1f} MB)" if ok else "FAILED"
                    print(f"[{completed}/{n_total}] {filename}  ({span})  {tag}")
                if ok:
                    n_done += 1
                    total_mb += size_mb
                    manifest_entries.append(
                        {"file": filename, "status": "downloaded", "size_mb": round(size_mb, 2)}
                    )
                else:
                    n_failed += 1
                    manifest_entries.append(
                        {"file": filename, "status": "failed", "size_mb": 0.0}
                    )
        except KeyboardInterrupt:
            print("\n  Interrupted. Cancelling queued chunks ...", file=sys.stderr)
            for f in futures:
                f.cancel()
            print("  NOTE: requests already submitted keep running on the CDS and",
                  file=sys.stderr)
            print("  hold queue slots. Clear them at "
                  "https://cds.climate.copernicus.eu/requests", file=sys.stderr)
            raise

    elapsed_min = (time.monotonic() - t_start) / 60.0

    # --- Manifest ------------------------------------------------------------
    # Records exactly what was requested so a run can be reproduced or audited
    # later without re-deriving the arguments.
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "frequency": args.frequency,
        "daily_statistic": args.daily_statistic if args.frequency == "daily" else None,
        "monthly_product": args.monthly_product if args.frequency == "monthly" else None,
        "region": region_label,
        "region_description": region.description,
        "area_north_west_south_east_deg": region.as_area(),
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "hours_utc": ALL_HOURS,
        "variable_set": args.var_set,
        "variables": [
            {"cds_name": v.cds_name, "short_name": v.short_name, "units": v.units, "role": v.role}
            for v in VARIABLE_SETS[args.var_set]
        ],
        "grid_deg": args.grid or NATIVE_GRID_DEG,
        "chunking": chunk,
        "sign_convention": (
            "ERA5 surface fluxes are positive DOWNWARD. Sledd et al. (2025) Eq. (1) "
            "defines SH and LH positive UPWARD, so SH = -msshf and LH = -mslhf. "
            "See seb_terms.py."
        ),
        "reference": "Sledd et al. (2025), JGR Atmos, 130, e2024JD042578",
        "files": manifest_entries,
    }
    manifest_path = run_dir / f"manifest_{args.start:%Y%m%d}_{args.end:%Y%m%d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- Summary -------------------------------------------------------------
    print()
    print("=" * 72)
    print(
        f"Downloaded {n_done}, skipped {n_skipped}, failed {n_failed} "
        f"of {len(chunks)} chunks in {elapsed_min:.1f} min"
    )
    print(f"Total new data: {total_mb / 1024:.2f} GB")
    print(f"Output dir    : {run_dir}")
    print(f"Manifest      : {manifest_path}")
    print("=" * 72)

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
