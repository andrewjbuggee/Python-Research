"""Lazy, region-subset access to ERA5 on the NSF NCAR public S3 bucket.

    https://registry.opendata.aws/nsf-ncar-era5/
    s3://nsf-ncar-era5   region us-west-2   anonymous (no AWS account needed to READ)
    DOI 10.5065/BH6N-5N20    (Hersbach et al. 2020, doi:10.1002/qj.3803)

What this module returns
------------------------
``open_dataset(...)`` builds an :class:`xarray.Dataset` whose data variables are
LAZY: nothing is read from S3 until a caller indexes a variable and calls
``.load()`` / ``.values``. The Dataset is shaped exactly like a file written by
``download_era5_seb.py`` -- dims ``(valid_time, latitude, longitude)``, latitude
descending, longitude ascending on [-180, 180), canonical short names, float32,
NaN over land -- so it can be handed to every function in ``seb_analysis_common``
/ ``surface_classification`` unchanged. The consumers in this project stream
through ``iter_time_blocks``, which does ``ds[vars].isel(valid_time=slice(i0,
i1)).load()``; with this backend that reads exactly the block requested and
only the variables named.

Why not ``xr.open_mfdataset`` over the bucket?  Opening one HDF5 file over HTTP
costs 0.6-2 s of metadata round trips from a laptop, and the archive has one
file per variable per month (surface analysis), per half-month (forecast
fluxes) or per DAY (pressure levels): one cold season of the Ocean Visions
figure set touches ~1,500 files. Here the time axis of every file is PREDICTED
from its name (the archive is strictly regular) and VERIFIED against the file's
own time variable the first time that file is opened, so building the lazy
Dataset costs only S3 listings, which are cached on disk.

Why not read the data through h5py?  h5py holds a global lock around every
libhdf5 call -- including the gzip decode of a chunk and the Python callbacks
that fetch bytes from S3 -- so threads overlap almost nothing, and one
connection moves ~2 MB/s from a laptop. Instead this module uses h5py ONLY for
metadata (dataset shape/chunking/filters and each chunk's byte offset, all
cached on disk per file) and fetches the compressed chunks itself with
concurrent HTTP range requests (``fsspec.cat_ranges``), then inflates them with
``zlib`` -- both release the GIL. Measured on the flux files from a laptop:
8 initialisations in 9.4 s this way against 78 s through h5py, values
bit-identical. The decoded regional slab of every HDF5 chunk is cached on disk
(see "Local cache"), so a second pass over the same data -- the OV notebook
makes four or five -- reads nothing from S3 at all.

Bucket layout (verified against the live bucket, 2026-09-18)
-------------------------------------------------------------
Coverage 1940-01 to 2026-06 (analysis) / 2026-05 (forecast); monthly updates
with a 3-4 month lag.

group                     file span   time layout                      HDF5 chunk
------------------------  ----------  -------------------------------  -------------------
e5.oper.an.sfc            month       hourly ``time``                  (27, 139, 277)
e5.oper.an.vinteg         month       hourly ``time``                  (27, 139, 277)
e5.oper.an.pl             day         hourly ``time`` x 37 ``level``   (1, 37, 721, 1440)
e5.oper.fc.sfc.meanflux   half-month  (forecast_initial_time,          (1, 12, 721, 1440)
e5.oper.fc.sfc.instan                  forecast_hour) 2-D grid
e5.oper.fc.sfc.accumu
e5.oper.invariant/197901  once        no time axis (lsm, z, ...)       -

All chunks are gzip(level 1) + byte-shuffle, float32, land as the fill value
9.999e+20. Chunk shape is the operative cost: a surface-analysis chunk is a
27 h x 139 x 277 tile, so a regional window pulls a few ~0.5 MB tiles per day,
while a flux or pressure-level chunk is the WHOLE GLOBE for one initialisation
/ one hour (~18 MB / 8-65 MB compressed) regardless of the window. Run large
pulls on EC2 in us-west-2 (see ``remote/``), where the transfer is in-region.

Forecast (flux) time semantics
------------------------------
ERA5 forecasts are initialised at 06 and 18 UTC and archived for forecast hours
1..12. A mean-rate field stamped at forecast hour h is the mean over the hour
(h-1, h] (ECMWF, ERA5 data documentation, "Mean rates/fluxes and
accumulations", https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation),
so its valid time is ``forecast_initial_time + forecast_hour``. That is also
how the CDS stamps ``valid_time`` for the ``avg_*`` fields the local archive
was built from; verified bit-for-bit against the local Barrow files
(``verify_against_local.py``). Consequence: the first six hours of any month
(00-06 UTC on the 1st) come from the SECOND half-month file of the PREVIOUS
month, so this module always lists one extra half-month file ahead of a
request. Adjacent initialisations tile time exactly -- no gaps, no duplicates.

Not in the bucket: ``tcslw`` (total column supercooled liquid water) and ``tp``
as an hourly accumulation. ``tp`` is synthesised here from ``mtpr`` (mean total
precipitation rate, kg m-2 s-1, itself the mean over the preceding hour):
1 kg m-2 of water is 1 mm depth, so ``tp_m = mtpr * 3600 s / 1000 = mtpr * 3.6``.
Against the local CDS ``tp`` this agrees to 1.5e-6 m per hour (GRIB packing
precision; 1.5% of the 0.1 mm/hr precipitation-filter threshold).

Local cache
-----------
``ERA5_S3_CACHE`` (default ``~/.cache/era5_s3``) holds:
  listings/   the S3 directory listing of every month touched
  meta/       per file: shape, chunking, filters, CF attrs, chunk byte offsets
  chunks/     per (file, HDF5 chunk, spatial window[, levels]): the decoded
              regional slab as float32 ``.npy``
The chunk cache is what turns the bucket into an on-demand local archive: only
what the analysis actually touched is stored, at the window's size (a Barrow
cold season of every SEB variable is ~1 GB), and any later run over the same
box and period costs no S3 traffic. Point ``ERA5_S3_CACHE`` at the external
drive for a large box, or set ``ERA5_S3_CHUNK_CACHE=off`` to disable that layer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
import threading
import time as _time
import warnings
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

# ----------------------------------------------------------------------------
# Bucket constants
# ----------------------------------------------------------------------------
BUCKET = "nsf-ncar-era5"
AWS_REGION = "us-west-2"

# The ERA5 0.25 deg regular grid as stored in every file of the bucket:
# latitude 90 -> -90 (721 rows, descending); longitude 0 -> 359.75 (1440 cols).
# Hard-coded so that building a Dataset needs no file open; verified against
# the file's own coordinate arrays the first time each file is opened.
N_LAT, N_LON = 721, 1440
GRID_DEG = 0.25
LAT_ALL_DEG = 90.0 - GRID_DEG * np.arange(N_LAT)
LON_ALL_DEG = GRID_DEG * np.arange(N_LON)

# Time variables are int32 "hours since 1900-01-01 00:00:00" (checked on open).
TIME_EPOCH = np.datetime64("1900-01-01T00:00:00", "h")

# Forecast archive geometry: initialisations at 06 and 18 UTC, hours 1..12.
FC_INIT_STEP_H = 12
FC_HOURS = np.arange(1, 13)

# Group layouts
LAYOUT_ANALYSIS = "analysis"        # (time, lat, lon), monthly files
LAYOUT_ANALYSIS_PL = "analysis_pl"  # (time, level, lat, lon), daily files
LAYOUT_FORECAST = "forecast"        # (init, fhour, lat, lon), half-month files
LAYOUT_INVARIANT = "invariant"      # (lat, lon) after squeezing a length-1 time

GROUP_LAYOUT = {
    "e5.oper.an.sfc": LAYOUT_ANALYSIS,
    "e5.oper.an.vinteg": LAYOUT_ANALYSIS,
    "e5.oper.an.pl": LAYOUT_ANALYSIS_PL,
    "e5.oper.fc.sfc.meanflux": LAYOUT_FORECAST,
    "e5.oper.fc.sfc.instan": LAYOUT_FORECAST,
    "e5.oper.fc.sfc.accumu": LAYOUT_FORECAST,
    "e5.oper.invariant": LAYOUT_INVARIANT,
}
INVARIANT_MONTH = (1979, 1)

# All 37 ERA5 pressure levels, ascending as stored in the files (hPa).
LEVELS_ALL_HPA = np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
                           225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                           775, 800, 825, 850, 875, 900, 925, 950, 975, 1000], dtype="float64")


# ----------------------------------------------------------------------------
# Variable registry: canonical project name -> (group, NCAR/GRIB short name)
# ----------------------------------------------------------------------------
# Canonical names are the ones ``download_era5_seb.py`` writes (its
# normalise_names table) and every analysis module reads. NCAR keeps the
# original ECMWF GRIB short names, so sea ice cover is ``ci`` not ``siconc``
# and 2 m temperature is ``2t`` not ``t2m``. Units and long names follow the
# ECMWF parameter database (https://codes.ecmwf.int/grib/param-db/) and match
# the attributes the CDS writes into the local files.
@dataclass(frozen=True)
class VarSpec:
    group: str
    ncar: str
    units: str
    long_name: str


NCAR_VARS: dict[str, VarSpec] = {
    # --- time-mean surface fluxes (forecast, half-month files), W m-2 -------
    # ERA5 convention: ALL surface fluxes positive DOWNWARD (into the surface).
    "msshf": VarSpec("e5.oper.fc.sfc.meanflux", "msshf", "W m**-2", "Time-mean surface sensible heat flux"),
    "mslhf": VarSpec("e5.oper.fc.sfc.meanflux", "mslhf", "W m**-2", "Time-mean surface latent heat flux"),
    "msnlwrf": VarSpec("e5.oper.fc.sfc.meanflux", "msnlwrf", "W m**-2", "Time-mean surface net long-wave radiation flux"),
    "msnswrf": VarSpec("e5.oper.fc.sfc.meanflux", "msnswrf", "W m**-2", "Time-mean surface net short-wave radiation flux"),
    "msdwlwrf": VarSpec("e5.oper.fc.sfc.meanflux", "msdwlwrf", "W m**-2", "Time-mean surface downward long-wave radiation flux"),
    "msdwswrf": VarSpec("e5.oper.fc.sfc.meanflux", "msdwswrf", "W m**-2", "Time-mean surface downward short-wave radiation flux"),
    "msnlwrfcs": VarSpec("e5.oper.fc.sfc.meanflux", "msnlwrfcs", "W m**-2", "Time-mean surface net long-wave radiation flux, clear sky"),
    "msnswrfcs": VarSpec("e5.oper.fc.sfc.meanflux", "msnswrfcs", "W m**-2", "Time-mean surface net short-wave radiation flux, clear sky"),
    "msdwlwrfcs": VarSpec("e5.oper.fc.sfc.meanflux", "msdwlwrfcs", "W m**-2", "Time-mean surface downward long-wave radiation flux, clear sky"),
    "msdwswrfcs": VarSpec("e5.oper.fc.sfc.meanflux", "msdwswrfcs", "W m**-2", "Time-mean surface downward short-wave radiation flux, clear sky"),
    "mtpr": VarSpec("e5.oper.fc.sfc.meanflux", "mtpr", "kg m**-2 s**-1", "Mean total precipitation rate"),
    "msr": VarSpec("e5.oper.fc.sfc.meanflux", "msr", "kg m**-2 s**-1", "Mean snowfall rate"),
    "mtnlwrf": VarSpec("e5.oper.fc.sfc.meanflux", "mtnlwrf", "W m**-2", "Mean top net long-wave radiation flux"),
    "mtnswrf": VarSpec("e5.oper.fc.sfc.meanflux", "mtnswrf", "W m**-2", "Mean top net short-wave radiation flux"),
    "mtdwswrf": VarSpec("e5.oper.fc.sfc.meanflux", "mtdwswrf", "W m**-2", "Mean top downward short-wave radiation flux"),
    # --- surface analysis (monthly files) ---------------------------------
    "siconc": VarSpec("e5.oper.an.sfc", "ci", "(0 - 1)", "Sea ice area fraction"),
    "skt": VarSpec("e5.oper.an.sfc", "skt", "K", "Skin temperature"),
    "t2m": VarSpec("e5.oper.an.sfc", "2t", "K", "2 metre temperature"),
    "d2m": VarSpec("e5.oper.an.sfc", "2d", "K", "2 metre dewpoint temperature"),
    "u10": VarSpec("e5.oper.an.sfc", "10u", "m s**-1", "10 metre U wind component"),
    "v10": VarSpec("e5.oper.an.sfc", "10v", "m s**-1", "10 metre V wind component"),
    "sp": VarSpec("e5.oper.an.sfc", "sp", "Pa", "Surface pressure"),
    "msl": VarSpec("e5.oper.an.sfc", "msl", "Pa", "Mean sea level pressure"),
    "tcc": VarSpec("e5.oper.an.sfc", "tcc", "(0 - 1)", "Total cloud cover"),
    "lcc": VarSpec("e5.oper.an.sfc", "lcc", "(0 - 1)", "Low cloud cover"),
    "mcc": VarSpec("e5.oper.an.sfc", "mcc", "(0 - 1)", "Medium cloud cover"),
    "hcc": VarSpec("e5.oper.an.sfc", "hcc", "(0 - 1)", "High cloud cover"),
    "tciw": VarSpec("e5.oper.an.sfc", "tciw", "kg m**-2", "Total column cloud ice water"),
    "tclw": VarSpec("e5.oper.an.sfc", "tclw", "kg m**-2", "Total column cloud liquid water"),
    "tcsw": VarSpec("e5.oper.an.sfc", "tcsw", "kg m**-2", "Total column snow water"),
    "tcrw": VarSpec("e5.oper.an.sfc", "tcrw", "kg m**-2", "Total column rain water"),
    "tcwv": VarSpec("e5.oper.an.sfc", "tcwv", "kg m**-2", "Total column water vapour"),
    "tcw": VarSpec("e5.oper.an.sfc", "tcw", "kg m**-2", "Total column water"),
    "fal": VarSpec("e5.oper.an.sfc", "fal", "(0 - 1)", "Forecast albedo"),
    "sst": VarSpec("e5.oper.an.sfc", "sstk", "K", "Sea surface temperature"),
    "istl1": VarSpec("e5.oper.an.sfc", "istl1", "K", "Ice temperature layer 1"),
    "istl2": VarSpec("e5.oper.an.sfc", "istl2", "K", "Ice temperature layer 2"),
    "istl3": VarSpec("e5.oper.an.sfc", "istl3", "K", "Ice temperature layer 3"),
    "istl4": VarSpec("e5.oper.an.sfc", "istl4", "K", "Ice temperature layer 4"),
    "blh": VarSpec("e5.oper.an.sfc", "blh", "m", "Boundary layer height"),
    "sd": VarSpec("e5.oper.an.sfc", "sd", "m of water equivalent", "Snow depth"),
    "rsn": VarSpec("e5.oper.an.sfc", "rsn", "kg m**-3", "Snow density"),
    "cape": VarSpec("e5.oper.an.sfc", "cape", "J kg**-1", "Convective available potential energy"),
    # --- instantaneous forecast surface (half-month files) ----------------
    "cbh": VarSpec("e5.oper.fc.sfc.instan", "cbh", "m", "Cloud base height"),
    "zust": VarSpec("e5.oper.fc.sfc.instan", "zust", "m s**-1", "Friction velocity"),
    # --- pressure levels (daily files, 37 levels) --------------------------
    "clwc": VarSpec("e5.oper.an.pl", "clwc", "kg kg**-1", "Specific cloud liquid water content"),
    "ciwc": VarSpec("e5.oper.an.pl", "ciwc", "kg kg**-1", "Specific cloud ice water content"),
    "crwc": VarSpec("e5.oper.an.pl", "crwc", "kg kg**-1", "Specific rain water content"),
    "cswc": VarSpec("e5.oper.an.pl", "cswc", "kg kg**-1", "Specific snow water content"),
    "cc": VarSpec("e5.oper.an.pl", "cc", "(0 - 1)", "Fraction of cloud cover"),
    "t": VarSpec("e5.oper.an.pl", "t", "K", "Temperature"),
    "q": VarSpec("e5.oper.an.pl", "q", "kg kg**-1", "Specific humidity"),
    "r": VarSpec("e5.oper.an.pl", "r", "%", "Relative humidity"),
    "u": VarSpec("e5.oper.an.pl", "u", "m s**-1", "U component of wind"),
    "v": VarSpec("e5.oper.an.pl", "v", "m s**-1", "V component of wind"),
    "w": VarSpec("e5.oper.an.pl", "w", "Pa s**-1", "Vertical velocity"),
    "z": VarSpec("e5.oper.an.pl", "z", "m**2 s**-2", "Geopotential"),
    # --- invariants ---------------------------------------------------------
    "lsm": VarSpec("e5.oper.invariant", "lsm", "(0 - 1)", "Land-sea mask"),
    "z_sfc": VarSpec("e5.oper.invariant", "z", "m**2 s**-2", "Geopotential (surface)"),
}

# Canonical names that are NOT in the bucket but can be built from one that is:
# name -> (source canonical, multiplicative factor, units, long_name, note)
DERIVED_VARS: dict[str, tuple[str, float, str, str, str]] = {
    # 1 kg m-2 = 1 mm of water; mean rate over the preceding hour x 3600 s,
    # expressed in metres like the CDS hourly accumulation. Identical quantity.
    "tp": ("mtpr", 3.6, "m", "Total precipitation",
           "synthesised as mtpr [kg m-2 s-1] x 3.6 (= x 3600 s / 1000 kg m-3)"),
}

# Canonical names in the project's CDS variable sets that have no S3 source.
UNAVAILABLE_ON_S3 = {"tcslw"}

PRESSURE_LEVEL_VARS = {k for k, v in NCAR_VARS.items() if v.group == "e5.oper.an.pl"}

# Every single-level variable the Ocean Visions figure modules read
# (plot_lwp_histogram_by_surface_class, cloud_spatial_extent, map_liquid_hours,
# plot_dlr_by_phase, fit_cloud_thresholds, turbulent_flux_response), plus the
# rest of the downloader's 'recommended' set that the bucket carries. Declaring
# a variable is free: the analysis modules stream only the ones they read.
SEB_STANDARD: tuple[str, ...] = (
    "msdwlwrf", "msdwlwrfcs", "msnlwrf", "msnlwrfcs", "msdwswrf", "msdwswrfcs",
    "msnswrf", "msnswrfcs", "msshf", "mslhf", "tp",
    "siconc", "skt", "t2m", "d2m", "u10", "v10", "sp",
    "tcc", "lcc", "mcc", "hcc", "tclw", "tciw", "tcrw", "tcsw", "tcwv",
    "fal", "sst", "istl1", "istl2", "istl3", "istl4",
)


# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------
DEBUG = bool(os.environ.get("ERA5_S3_DEBUG"))

# fsspec block size for the h5py METADATA reads. HDF5 metadata is a handful of
# small scattered reads, so this only sets how much is over-fetched around
# them; 1 MiB "bytes" caching measured best (open ~0.6 s from a laptop).
FS_BLOCK_SIZE = 2 ** 20
FS_CACHE_TYPE = "bytes"

CACHE_DIR = Path(os.environ.get("ERA5_S3_CACHE", Path.home() / ".cache" / "era5_s3")).expanduser()
CHUNK_CACHE_ON = os.environ.get("ERA5_S3_CHUNK_CACHE", "on").lower() not in ("off", "0", "false", "no")

# Files of a group are re-listed when the month is this recent, because NCAR
# appends to the current months as ERA5T is replaced by final ERA5.
RELIST_IF_YOUNGER_THAN_DAYS = 200

# Transient S3/network failures are retried this many times with backoff.
FETCH_RETRIES = 4

# What to do about requested hours the bucket does not have (the forecast
# groups lag the analysis groups by a month; a request into the future):
# "error" (default) or "nan". ``ERA5_S3_MISSING=nan`` for a run that must go
# on regardless; the adapter forwards it to every open_dataset call.
MISSING_DEFAULT = os.environ.get("ERA5_S3_MISSING", "error").lower()


def n_workers() -> int:
    """Threads that fetch+decode concurrently. ``ERA5_S3_WORKERS`` overrides.

    Each thread issues its own chunk-range requests concurrently, so the
    default of 8 keeps tens of HTTP ranges in flight. Raise it on an EC2
    instance, where decompression rather than the link becomes the limit.
    """
    env = os.environ.get("ERA5_S3_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(8, os.cpu_count() or 4))


# ----------------------------------------------------------------------------
# Filesystem + listings
# ----------------------------------------------------------------------------
_FS = None
_FS_LOCK = threading.Lock()


def filesystem():
    """Anonymous s3fs handle, one per interpreter. Imported lazily."""
    global _FS
    with _FS_LOCK:
        if _FS is None:
            import fsspec

            _FS = fsspec.filesystem("s3", anon=True, default_block_size=FS_BLOCK_SIZE)
        return _FS


# e5.oper.an.sfc.128_031_ci.ll025sc.2024100100_2024103123.nc
# e5.oper.an.pl.128_131_u.ll025uv.2024100100_2024100123.nc      <- 'uv' grid tag
# e5.oper.fc.sfc.meanflux.235_036_msdwlwrf.ll025sc.2024100106_2024101606.nc
_NAME_RE = re.compile(
    r"\.(?P<table>\d{3})_(?P<num>\d{3})_(?P<short>[0-9a-z]+)\.ll025(?P<grid>[a-z]{2})"
    r"\.(?P<t0>\d{10})_(?P<t1>\d{10})\.nc$"
)


@dataclass(frozen=True)
class S3File:
    """One object in the bucket and what its name says it holds."""

    key: str            # full "bucket/group/YYYYMM/name.nc"
    group: str
    short: str          # NCAR short name, e.g. 'ci'
    param_id: int       # ECMWF paramId (table 128 -> num; else table*1000+num)
    t0: np.datetime64   # first stamp in the name, hour precision
    t1: np.datetime64   # last stamp in the name, hour precision
    size_bytes: int

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def layout(self) -> str:
        return GROUP_LAYOUT[self.group]


def _parse_stamp(s: str) -> np.datetime64:
    return np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}", "h")


_LISTING_MEM: dict[tuple[str, int, int], tuple[S3File, ...]] = {}
_LISTING_LOCK = threading.Lock()


def list_month(group: str, year: int, month: int, refresh: bool = False) -> tuple[S3File, ...]:
    """Every file of ``group`` for one month, from memory, disk cache, or S3."""
    key = (group, year, month)
    with _LISTING_LOCK:
        if key in _LISTING_MEM and not refresh:
            return _LISTING_MEM[key]

    cache = CACHE_DIR / "listings" / f"{group}_{year:04d}{month:02d}.json"
    recent = (date.today() - date(year, month, 1)).days < RELIST_IF_YOUNGER_THAN_DAYS
    entries = None
    if cache.exists() and not refresh and not recent:
        try:
            entries = json.loads(cache.read_text())
        except json.JSONDecodeError:
            entries = None
    if entries is None:
        fs = filesystem()
        prefix = f"{BUCKET}/{group}/{year:04d}{month:02d}/"
        try:
            raw = fs.ls(prefix, detail=True)
        except FileNotFoundError:
            raw = []
        entries = [{"name": r["name"], "size": int(r.get("size") or 0)} for r in raw]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(entries))

    files = []
    for e in entries:
        m = _NAME_RE.search(e["name"])
        if not m:
            continue
        table, num = int(m["table"]), int(m["num"])
        files.append(S3File(
            key=e["name"], group=group, short=m["short"],
            param_id=num if table == 128 else table * 1000 + num,
            t0=_parse_stamp(m["t0"]), t1=_parse_stamp(m["t1"]),
            size_bytes=e["size"],
        ))
    out = tuple(sorted(files, key=lambda f: (f.short, f.t0)))
    with _LISTING_LOCK:
        _LISTING_MEM[key] = out
    return out


def _prev_month(ym: tuple[int, int]) -> tuple[int, int]:
    y, m = ym
    return (y - 1, 12) if m == 1 else (y, m - 1)


def files_for(canonical: str, months: Iterable[tuple[int, int]]) -> list[S3File]:
    """All files holding ``canonical`` over ``months``, in time order.

    For forecast groups the month BEFORE each requested month is listed too,
    because valid hours 00..06 UTC on the 1st live in the previous month's
    second half-month file (module docstring). Every month, not just the
    first: a request for eleven Oct-Mar seasons has eleven such boundaries.
    """
    spec = NCAR_VARS[canonical]
    months = list(months)
    if spec.group == "e5.oper.invariant":
        months = [INVARIANT_MONTH]
    elif GROUP_LAYOUT[spec.group] == LAYOUT_FORECAST and months:
        months = sorted(set(months) | {_prev_month(m) for m in months})
    seen, out = set(), []
    for y, m in months:
        for f in list_month(spec.group, y, m):
            if f.short == spec.ncar and f.key not in seen:
                seen.add(f.key)
                out.append(f)
    return sorted(out, key=lambda f: f.t0)


# ----------------------------------------------------------------------------
# Predicted per-file time axes
# ----------------------------------------------------------------------------
def fc_n_init(f: S3File) -> int:
    """Initialisations in a forecast file: the name's t1 is the NEXT file's first."""
    return int((f.t1 - f.t0).astype(int)) // FC_INIT_STEP_H


def predicted_times(f: S3File) -> np.ndarray:
    """Valid times a file holds, from its name alone (``datetime64[h]``).

    analysis  : every hour t0..t1 inclusive (monthly or daily file).
    forecast  : initialisations every 12 h in [t0, t1), each with forecast
                hours 1..12, flattened init-major so consecutive entries are
                consecutive valid hours (init 06 covers 07..18, init 18 covers
                19..06 of the next day).
    invariant : one dummy stamp.
    """
    layout = f.layout
    if layout in (LAYOUT_ANALYSIS, LAYOUT_ANALYSIS_PL):
        n = int((f.t1 - f.t0).astype(int)) + 1
        return f.t0 + np.arange(n).astype("timedelta64[h]")
    if layout == LAYOUT_FORECAST:
        inits = f.t0 + (np.arange(fc_n_init(f)) * FC_INIT_STEP_H).astype("timedelta64[h]")
        valid = inits[:, None] + FC_HOURS[None, :].astype("timedelta64[h]")
        return valid.reshape(-1)
    return np.array([f.t0])


def n_steps_stored(f: S3File) -> int:
    """Length of the file's leading (time or initialisation) axis."""
    if f.layout == LAYOUT_FORECAST:
        return fc_n_init(f)
    if f.layout == LAYOUT_INVARIANT:
        return 1
    return int(predicted_times(f).size)


# ----------------------------------------------------------------------------
# Spatial window
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """Index bookkeeping for a lat/lon box on the stored 0..360 grid.

    ``lon_runs`` are contiguous ``slice`` objects in stored-index space, in the
    order their columns appear in the OUTPUT. Two runs means the output crosses
    stored column 0 (the Greenwich meridian): west=-10, east=+10 becomes stored
    columns 1400..1439 followed by 0..40.
    """

    lat_slice: slice
    lon_runs: tuple[slice, ...]
    lat_deg: np.ndarray       # descending, as stored and as the CDS writes them
    lon_deg: np.ndarray       # in [-180, 180), ascending unless the box crosses the dateline

    @property
    def n_lat(self) -> int:
        return int(self.lat_deg.size)

    @property
    def n_lon(self) -> int:
        return int(self.lon_deg.size)

    def tag(self) -> str:
        """Short stable id for cache paths."""
        return f"lat{self.lat_slice.start}-{self.lat_slice.stop}_lon" + "+".join(
            f"{r.start}-{r.stop}" for r in self.lon_runs)


def _to_180(lon: np.ndarray) -> np.ndarray:
    return np.where(lon >= 180.0, lon - 360.0, lon)


def make_window(north_deg: float, west_deg: float, south_deg: float, east_deg: float) -> Window:
    """Resolve a CDS-style ``[N, W, S, E]`` box (longitudes on [-180, 180]).

    The result reproduces the CDS's own selection: every grid point with
    ``south <= lat <= north`` and, for ``west <= east``, ``west <= lon <= east``
    in the -180..180 frame. A full circle ``[-180, 180]`` yields all 1440
    columns from -180 to 179.75, exactly as the CDS returns it (+180 is the
    same column as -180). A box with ``west > east`` crosses the dateline; its
    columns are returned contiguous in STORED order (e.g. 165 .. 179.75,
    -180 .. -160), non-monotonic in the -180..180 frame: label-based reindexing
    (``align_lsm_to_grid``) copes, ``.sel(longitude=slice(...))`` and the map
    code's ``lon.min()/max()`` do not, hence the warning.
    """
    if not (-90.0 <= south_deg < north_deg <= 90.0):
        raise ValueError(f"need -90 <= south < north <= 90, got S={south_deg} N={north_deg}")
    lat_hits = np.nonzero((LAT_ALL_DEG <= north_deg + 1e-9) & (LAT_ALL_DEG >= south_deg - 1e-9))[0]
    if lat_hits.size == 0:
        raise ValueError(f"no grid latitudes between {south_deg} and {north_deg}")
    lat_slice = slice(int(lat_hits[0]), int(lat_hits[-1]) + 1)

    lon180 = _to_180(LON_ALL_DEG)
    if west_deg <= east_deg:
        hit = (lon180 >= west_deg - 1e-9) & (lon180 <= east_deg + 1e-9)
        if east_deg >= 180.0 - 1e-9:
            hit |= lon180 <= -180.0 + 1e-9
        order = np.argsort(lon180, kind="stable")   # ascending in the -180..180 frame
        idx = order[hit[order]]
    else:
        hit = (lon180 >= west_deg - 1e-9) | (lon180 <= east_deg + 1e-9)
        idx = np.nonzero(hit)[0]                     # stored order: contiguous
        warnings.warn(
            f"box W={west_deg} > E={east_deg} crosses the dateline; the longitude "
            "coordinate is non-monotonic in the -180..180 frame", stacklevel=2,
        )
    if idx.size == 0:
        raise ValueError(f"no grid longitudes between {west_deg} and {east_deg}")

    runs: list[slice] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        runs.append(slice(start, prev + 1))
        start = prev = i
    runs.append(slice(start, prev + 1))

    return Window(
        lat_slice=lat_slice, lon_runs=tuple(runs),
        lat_deg=LAT_ALL_DEG[lat_slice].copy(),
        lon_deg=lon180[idx].astype("float64"),
    )


# ----------------------------------------------------------------------------
# Per-file metadata (h5py, cached on disk)
# ----------------------------------------------------------------------------
# h5py is used for METADATA ONLY, always under one lock: the fsspec file object
# behind it is not thread-safe, and libhdf5 is serialised by h5py's own global
# lock anyway.
_H5_LOCK = threading.RLock()
_H5_OPEN: "OrderedDict[str, object]" = OrderedDict()
_H5_MAX_OPEN = 32
_META_MEM: dict[str, "FileMeta"] = {}

# HDF5 filter ids (h5py.h5z.FILTER_*)
_F_DEFLATE, _F_SHUFFLE, _F_FLETCHER32 = 1, 2, 3


@dataclass
class FileMeta:
    """What is needed to fetch and decode any chunk of one file without h5py."""

    key: str
    size_bytes: int                # from the listing; part of the cache identity
    dset: str                      # data variable name inside the file
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: str
    filters: list[tuple[int, int]]  # (filter_id, flags) in pipeline (write) order
    fill: float | None
    scale: float | None
    offset: float | None
    chunk_offsets: dict[str, tuple[int, int, int] | None]  # "i,j,k" -> (byte_offset, nbytes, filter_mask); None = never written

    def to_json(self) -> dict:
        return {"key": self.key, "size_bytes": self.size_bytes, "dset": self.dset,
                "shape": list(self.shape), "chunks": list(self.chunks), "dtype": self.dtype,
                "filters": self.filters, "fill": self.fill, "scale": self.scale,
                "offset": self.offset, "chunk_offsets": self.chunk_offsets}

    @classmethod
    def from_json(cls, d: dict) -> "FileMeta":
        return cls(key=d["key"], size_bytes=d["size_bytes"], dset=d["dset"],
                   shape=tuple(d["shape"]), chunks=tuple(d["chunks"]), dtype=d["dtype"],
                   filters=[tuple(f) for f in d["filters"]], fill=d["fill"], scale=d["scale"],
                   offset=d["offset"],
                   chunk_offsets={k: (tuple(v) if v is not None else None)
                                  for k, v in d["chunk_offsets"].items()
                                  if v is None or len(v) == 3})   # drop pre-filter_mask entries


def _meta_path(f: S3File) -> Path:
    return CACHE_DIR / "meta" / f.group / f"{f.name}.json"


def _save_meta(meta: FileMeta, f: S3File) -> None:
    p = _meta_path(f)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(meta.to_json()))
    tmp.replace(p)


def _h5_open(key: str):
    """h5py.File for ``key`` from an LRU of open handles. Caller holds _H5_LOCK."""
    import h5py

    if key in _H5_OPEN:
        _H5_OPEN.move_to_end(key)
        return _H5_OPEN[key]
    while len(_H5_OPEN) >= _H5_MAX_OPEN:
        _, old = _H5_OPEN.popitem(last=False)
        try:
            old.close()
        except Exception:  # noqa: BLE001 - best effort on eviction
            pass
    fobj = filesystem().open(key, block_size=FS_BLOCK_SIZE, cache_type=FS_CACHE_TYPE)
    h = h5py.File(fobj, "r")
    _H5_OPEN[key] = h
    return h


def _data_name(h) -> str:
    """The one N-D data variable in a single-variable NCAR file."""
    skip = {"latitude", "longitude", "time", "level", "utc_date",
            "forecast_initial_time", "forecast_hour"}
    for n in h:
        obj = h[n]
        if n not in skip and getattr(obj, "shape", None) is not None and len(obj.shape) >= 2:
            return n
    raise KeyError("no data variable found")


def _attr_float(attrs, name: str) -> float | None:
    v = attrs.get(name)
    return None if v is None else float(np.asarray(v).ravel()[0])


def _verify_axes(h, f: S3File) -> None:
    """Check the file's own coordinates against the name-derived prediction.

    A failure means the archive is not as regular as assumed for this file,
    and the safe response is to stop rather than misalign every hour after it.
    """
    lat = np.asarray(h["latitude"][:], dtype="float64")
    lon = np.asarray(h["longitude"][:], dtype="float64")
    if lat.shape != (N_LAT,) or lon.shape != (N_LON,) \
            or not np.allclose(lat, LAT_ALL_DEG) or not np.allclose(lon, LON_ALL_DEG):
        raise ValueError(f"{f.name}: grid differs from the assumed 0.25 deg 721x1440 grid")
    if f.layout == LAYOUT_INVARIANT:
        return
    if f.layout == LAYOUT_ANALYSIS_PL:
        lev = np.asarray(h["level"][:], dtype="float64")
        if not np.allclose(lev, LEVELS_ALL_HPA):
            raise ValueError(f"{f.name}: level axis {lev} != the 37 standard levels")
    if f.layout == LAYOUT_FORECAST:
        tvar = h["forecast_initial_time"]
        fh = np.asarray(h["forecast_hour"][:], dtype="int64")
        if not np.array_equal(fh, FC_HOURS):
            raise ValueError(f"{f.name}: forecast_hour {fh} != {FC_HOURS}")
        step = FC_INIT_STEP_H
    else:
        tvar = h["time"]
        step = 1
    units = tvar.attrs.get("units", b"")
    units = units.decode() if isinstance(units, bytes) else str(units)
    if "1900-01-01" not in units or not units.startswith("hours"):
        raise ValueError(f"{f.name}: unexpected time units {units!r}")
    actual = TIME_EPOCH + np.asarray(tvar[:], dtype="int64").astype("timedelta64[h]")
    expected = f.t0 + (np.arange(n_steps_stored(f)) * step).astype("timedelta64[h]")
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(
            f"{f.name}: time axis differs from the name-derived prediction "
            f"(file {actual[0]}..{actual[-1]} n={actual.size}; predicted "
            f"{expected[0]}..{expected[-1]} n={expected.size})"
        )


def file_meta(f: S3File) -> FileMeta:
    """Metadata for ``f`` from memory, the disk cache, or one h5py open."""
    with _H5_LOCK:
        m = _META_MEM.get(f.key)
        if m is not None:
            return m
        p = _meta_path(f)
        if p.exists():
            try:
                m = FileMeta.from_json(json.loads(p.read_text()))
                if m.size_bytes == f.size_bytes:
                    _META_MEM[f.key] = m
                    return m
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        t0 = _time.time()
        h = _h5_open(f.key)
        name = _data_name(h)
        d = h[name]
        _verify_axes(h, f)
        plist = d.id.get_create_plist()
        filters = [tuple(int(x) for x in plist.get_filter(i)[:2]) for i in range(plist.get_nfilters())]
        for fid, _ in filters:
            if fid not in (_F_DEFLATE, _F_SHUFFLE, _F_FLETCHER32):
                raise NotImplementedError(f"{f.name}: HDF5 filter id {fid} not supported")
        chunks = tuple(int(c) for c in (d.chunks or d.shape))
        shape = tuple(int(s) for s in d.shape)
        if f.layout in (LAYOUT_FORECAST, LAYOUT_ANALYSIS_PL) and chunks[1] != shape[1]:
            raise NotImplementedError(f"{f.name}: expected axis 1 unchunked, got {chunks} of {shape}")
        m = FileMeta(
            key=f.key, size_bytes=f.size_bytes, dset=name, shape=shape, chunks=chunks,
            dtype=str(d.dtype), filters=filters,
            fill=_attr_float(d.attrs, "_FillValue") if "_FillValue" in d.attrs
            else _attr_float(d.attrs, "missing_value"),
            scale=_attr_float(d.attrs, "scale_factor"), offset=_attr_float(d.attrs, "add_offset"),
            chunk_offsets={},
        )
        _META_MEM[f.key] = m
        _save_meta(m, f)
        if DEBUG:
            print(f"    [meta] {f.name[:70]} opened+verified in {_time.time() - t0:.1f}s",
                  file=sys.stderr, flush=True)
        return m


def chunk_locations(f: S3File, meta: FileMeta, coords: list[tuple[int, ...]]) -> list[tuple[int, int, int] | None]:
    """Byte ranges of the chunks at grid coords ``coords`` (chunk-index units).

    Looked up through h5py's B-tree walk once and remembered in the file's
    metadata cache, so a later run over the same box never walks it again.
    """
    keys = [",".join(map(str, c)) for c in coords]
    missing = [c for c, k in zip(coords, keys) if k not in meta.chunk_offsets]
    if missing:
        with _H5_LOCK:
            h = _h5_open(f.key)
            d = h[meta.dset]
            for c in missing:
                k = ",".join(map(str, c))
                if k in meta.chunk_offsets:
                    continue
                start = tuple(ci * cs for ci, cs in zip(c, meta.chunks))
                try:
                    info = d.id.get_chunk_info_by_coord(start)
                    # filter_mask bit i set = filter i was skipped for this
                    # chunk (HDF5 does that for chunks that would not shrink).
                    loc = ((int(info.byte_offset), int(info.size), int(info.filter_mask))
                           if info.byte_offset is not None else None)
                except Exception as exc:  # noqa: BLE001 - h5py raises a generic error for absent chunks
                    msg = str(exc).lower()
                    if "not" in msg or "unable" in msg:
                        loc = None
                    else:
                        raise
                meta.chunk_offsets[k] = loc
            _save_meta(meta, f)
    return [meta.chunk_offsets[k] for k in keys]


# ----------------------------------------------------------------------------
# Chunk decode
# ----------------------------------------------------------------------------
def _unshuffle(raw: bytes, itemsize: int) -> np.ndarray:
    """Undo the HDF5 byte-shuffle filter (bytes grouped by significance)."""
    b = np.frombuffer(raw, dtype="u1")
    return np.ascontiguousarray(b.reshape(itemsize, -1).T).reshape(-1)


def decode_chunk(blob: bytes, meta: FileMeta, filter_mask: int = 0) -> np.ndarray:
    """Compressed chunk bytes -> array of ``meta.chunks`` shape, raw values.

    Filters are undone in reverse pipeline order. ``filter_mask`` bit i set
    means filter i was skipped for this chunk when written (HDF5 does that for
    chunks that would not shrink) -- honoured here, though never seen in this
    archive.
    """
    dt = np.dtype(meta.dtype)
    data = blob
    for i in range(len(meta.filters) - 1, -1, -1):
        fid, _flags = meta.filters[i]
        if filter_mask & (1 << i):
            continue
        if fid == _F_FLETCHER32:
            data = bytes(data)[:-4]          # checksum appended; not verified
        elif fid == _F_DEFLATE:
            data = zlib.decompress(bytes(data))
        elif fid == _F_SHUFFLE:
            data = _unshuffle(bytes(data), dt.itemsize).tobytes()
    arr = np.frombuffer(data, dtype=dt)
    n = int(np.prod(meta.chunks))
    if arr.size < n:
        raise ValueError(f"{meta.key}: decoded chunk has {arr.size} values, expected {n}")
    return arr[:n].reshape(meta.chunks)


def _cf_decode(values: np.ndarray, meta: FileMeta) -> np.ndarray:
    """Apply CF ``_FillValue`` / ``scale_factor`` / ``add_offset`` by hand.

    netCDF4 and xarray do this automatically; a raw chunk decode does not.
    These files store land as 9.999e+20, so skipping it turns every land cell
    into a finite number and ``classify_cells`` (which uses "siconc is NaN" as
    its land test) silently absorbs the North Slope into the ocean.
    """
    out = np.array(values, dtype="float32", copy=True)
    if meta.fill is not None:
        out[np.isclose(out, np.float32(meta.fill), rtol=1e-6, atol=0.0)] = np.nan
    else:
        out[out > 1e19] = np.nan
    if meta.scale is not None:
        out *= np.float32(meta.scale)
    if meta.offset is not None:
        out += np.float32(meta.offset)
    return out


# ----------------------------------------------------------------------------
# Reading one file's window for a range of stored steps
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ReadSpec:
    """Everything one read task needs, independent of xarray."""

    f: S3File
    lo: int                        # stored-step range [lo, hi) on the leading axis
    hi: int                        # (time steps, or initialisations for forecast files)
    window: Window
    level_idx: tuple[int, ...] | None   # stored level indices, ascending, or None
    factor: float


def _chunk_cache_path(f: S3File, chunk_coord: tuple[int, ...], window: Window,
                      level_idx: tuple[int, ...] | None) -> Path:
    lev = "all" if level_idx is None else hashlib.sha1(
        ",".join(map(str, level_idx)).encode()).hexdigest()[:10]
    cid = "_".join(map(str, chunk_coord))
    return (CACHE_DIR / "chunks" / f.group / f"{f.name}__{f.size_bytes}"
            / f"{window.tag()}__lev{lev}" / f"c{cid}.npy")


def _fetch_ranges(key: str, locs: list[tuple[int, int]]) -> list[bytes]:
    """Concurrent multi-range GET with retries on transient failures."""
    fs = filesystem()
    delay = 2.0
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            blobs = fs.cat_ranges([key] * len(locs), [a for a, _ in locs],
                                  [a + n for a, n in locs], on_error="return")
            blobs = list(blobs)
            bad = [b for b in blobs if isinstance(b, BaseException)]
            if not bad:
                short = [i for i, (b, (_, n)) in enumerate(zip(blobs, locs)) if len(b) != n]
                if not short:
                    return blobs
                raise OSError(f"{len(short)} range(s) returned the wrong length")
            raise bad[0]
        except Exception as exc:  # noqa: BLE001 - network layer raises many types
            if attempt == FETCH_RETRIES:
                raise
            print(f"  S3 fetch of {key.rsplit('/', 1)[-1][:50]} failed ({type(exc).__name__}: "
                  f"{str(exc)[:80]}); retry {attempt}/{FETCH_RETRIES - 1} in {delay:.0f}s",
                  file=sys.stderr, flush=True)
            _time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _lon_cols_in_chunk(window: Window, a_lon: int, c_lon: int) -> np.ndarray:
    """Stored columns of the window inside one chunk column, chunk-relative,
    in the order the window's runs list them."""
    cols = []
    for run in window.lon_runs:
        lo_, hi_ = max(run.start, a_lon), min(run.stop, a_lon + c_lon)
        if lo_ < hi_:
            cols.append(np.arange(lo_ - a_lon, hi_ - a_lon))
    return np.concatenate(cols) if cols else np.arange(0)


def _slab_from_chunk(full: np.ndarray, coord: tuple[int, ...], chunks: tuple[int, ...],
                     spec: ReadSpec) -> np.ndarray:
    """The part of a decoded chunk that this window (and level set) needs.

    Leading axes (time/init, forecast hour) are kept whole so the cached slab
    serves any step range; lat and lon are cut to the window; the level axis is
    cut to the requested stored indices. This is what the chunk cache stores.
    """
    ndim = full.ndim
    lat_ax, lon_ax = ndim - 2, ndim - 1
    a_lat = coord[lat_ax] * chunks[lat_ax]
    lat_sl = slice(max(spec.window.lat_slice.start, a_lat) - a_lat,
                   min(spec.window.lat_slice.stop, a_lat + chunks[lat_ax]) - a_lat)
    lon_idx = _lon_cols_in_chunk(spec.window, coord[lon_ax] * chunks[lon_ax], chunks[lon_ax])
    sub = full[..., lat_sl, :][..., lon_idx]
    if spec.f.layout == LAYOUT_ANALYSIS_PL and spec.level_idx is not None:
        sub = sub[:, list(spec.level_idx)]
    return np.ascontiguousarray(sub)


def read_window(spec: ReadSpec) -> np.ndarray:
    """Decoded window for stored steps ``[lo, hi)`` of one file.

    Returns float32 with NaN fill, shaped ``(hi-lo, n_lat, n_lon)`` for
    analysis files, ``(hi-lo, 12, n_lat, n_lon)`` for forecast files (one row
    per initialisation), ``(hi-lo, n_lev, n_lat, n_lon)`` for pressure-level
    files (levels in ascending stored order) and ``(n_lat, n_lon)`` for
    invariants.

    Work is organised per HDF5 chunk: each chunk's regional slab is looked up
    in the on-disk chunk cache; the missing ones are fetched with ONE
    concurrent multi-range request, decoded, and cached.
    """
    f, meta, w = spec.f, file_meta(spec.f), spec.window
    shape, chunks = meta.shape, meta.chunks
    layout = f.layout
    ndim = len(shape)

    # Stored-index selection per axis. Axis 1 of forecast / pressure-level
    # files is unchunked (checked in file_meta) so it is always one chunk.
    if layout == LAYOUT_INVARIANT:
        lead = (slice(0, 1),) if ndim == 3 else ()
    else:
        lead = (slice(spec.lo, spec.hi),)
    mid = (slice(0, shape[1]),) if layout in (LAYOUT_FORECAST, LAYOUT_ANALYSIS_PL) else ()
    n_lev_out = (len(spec.level_idx) if (layout == LAYOUT_ANALYSIS_PL and spec.level_idx is not None)
                 else (shape[1] if mid else None))
    out_shape = [s.stop - s.start for s in lead]
    if mid:
        out_shape.append(n_lev_out)
    out_shape += [w.n_lat, w.n_lon]
    out = np.full(out_shape, np.nan, dtype="float32")

    def crange(s: slice, cs: int) -> range:
        return range(s.start // cs, (s.stop - 1) // cs + 1)

    # Enumerate intersecting chunks per lon run; dst is where the piece lands
    # in ``out`` (lon offset accumulates across runs).
    jobs: list[tuple[tuple[int, ...], tuple, tuple]] = []
    col0 = 0
    for run in w.lon_runs:
        sel = list(lead) + list(mid) + [w.lat_slice, run]
        for coord in itertools.product(*[crange(s, cs) for s, cs in zip(sel, chunks)]):
            src, dst = [], []
            for ax, (ci, cs, s) in enumerate(zip(coord, chunks, sel)):
                a = ci * cs
                lo_, hi_ = max(s.start, a), min(s.stop, a + cs)
                src.append(slice(lo_ - a, hi_ - a))
                base = (s.start - col0) if ax == len(sel) - 1 else s.start
                dst.append(slice(lo_ - base, hi_ - base))
            jobs.append((coord, tuple(src), tuple(dst)))
        col0 += run.stop - run.start

    # Slabs from cache; the rest in one concurrent fetch.
    slabs: dict[tuple[int, ...], np.ndarray | None] = {}
    to_fetch: list[tuple[int, ...]] = []
    for coord, _, _ in jobs:
        if coord in slabs or coord in to_fetch:
            continue
        if CHUNK_CACHE_ON:
            p = _chunk_cache_path(f, coord, w, spec.level_idx)
            if p.exists():
                try:
                    slabs[coord] = np.load(p)
                    continue
                except (OSError, ValueError, EOFError):
                    # A truncated file (crash or power loss mid-write on a
                    # volume without fsync) must not poison every later run.
                    try:
                        p.unlink()
                    except OSError:
                        pass
        to_fetch.append(coord)
    n_cached = len(slabs)

    if to_fetch:
        locs = chunk_locations(f, meta, to_fetch)
        present = [(c, l) for c, l in zip(to_fetch, locs) if l is not None]
        t0 = _time.time()
        blobs = _fetch_ranges(f.key, [(l[0], l[1]) for _, l in present]) if present else []
        t_fetch = _time.time() - t0
        for (coord, loc), blob in zip(present, blobs):
            slab = _slab_from_chunk(_cf_decode(decode_chunk(blob, meta, loc[2]), meta), coord, chunks, spec)
            slabs[coord] = slab
            if CHUNK_CACHE_ON:
                p = _chunk_cache_path(f, coord, w, spec.level_idx)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_name(p.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, slab)
                tmp.replace(p)
        for coord, loc in zip(to_fetch, locs):
            if loc is None:
                slabs[coord] = None      # never written: HDF5 fill value = missing
        if DEBUG:
            n_bytes = sum(len(b) for b in blobs)
            print(f"    [read] {f.name[:58]:<58} steps {spec.lo:>4}-{spec.hi:<4} "
                  f"{len(present):>3} chunks {n_bytes / 1e6:7.1f} MB in {t_fetch:5.1f}s"
                  f"  (+{n_cached} cached)", file=sys.stderr, flush=True)

    # Assemble. A slab is the chunk's window subset: leading axes whole, lat
    # rows from the window's first row inside the chunk, lon columns as the
    # window's runs list them, levels = requested set.
    for coord, src, dst in jobs:
        slab = slabs.get(coord)
        if slab is None:
            continue
        lat_ax, lon_ax = slab.ndim - 2, slab.ndim - 1
        a_lat = coord[lat_ax] * chunks[lat_ax]
        slab_lat0 = max(w.lat_slice.start, a_lat) - a_lat
        lat_sl = slice(src[lat_ax].start - slab_lat0, src[lat_ax].stop - slab_lat0)
        all_cols = _lon_cols_in_chunk(w, coord[lon_ax] * chunks[lon_ax], chunks[lon_ax])
        # The slab lists columns run by run, so all_cols is unsorted whenever
        # two runs meet the same chunk (a whole-globe chunk with a full-circle
        # box): locate this run's columns through an inverse permutation.
        order_c = np.argsort(all_cols, kind="stable")
        want = np.arange(src[lon_ax].start, src[lon_ax].stop)
        pos = order_c[np.searchsorted(all_cols[order_c], want)]
        piece = slab[..., lat_sl, :][..., pos]
        if slab.ndim >= 3:
            piece = piece[src[0]]            # leading axis (time / init)
        if layout == LAYOUT_FORECAST:
            piece = piece[:, src[1]]         # forecast hours (always all 12)
        # pressure levels: slab already holds exactly the requested levels
        full_dst = tuple(dst[:len(lead)]) + ((slice(None),) if mid else ()) + tuple(dst[-2:])
        out[full_dst] = piece
    if spec.factor != 1.0:
        out *= np.float32(spec.factor)
    return out


# ----------------------------------------------------------------------------
# Thread pool
# ----------------------------------------------------------------------------
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=n_workers(), thread_name_prefix="era5_s3")
        return _POOL


def shutdown_pool() -> None:
    """Stop the reader threads and close cached HDF5 metadata handles."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
            _POOL = None
    with _H5_LOCK:
        while _H5_OPEN:
            _, h = _H5_OPEN.popitem()
            try:
                h.close()
            except Exception:  # noqa: BLE001
                pass


def _run(specs: list[ReadSpec]) -> list[np.ndarray]:
    if len(specs) <= 1 or n_workers() == 1:
        return [read_window(s) for s in specs]
    return list(_pool().map(read_window, specs))


# ----------------------------------------------------------------------------
# Lazy backend array
# ----------------------------------------------------------------------------
@dataclass
class VarIndex:
    """Where every step of one variable's global time axis lives on S3."""

    canonical: str
    files: list[S3File]
    file_of_step: np.ndarray      # int32 [n_time], -1 where no file covers it
    local_of_step: np.ndarray     # int32 [n_time], flat valid-time index inside that file
    factor: float = 1.0
    n_missing: int = 0


# Stored steps per read task: 4 analysis chunks, 4 initialisations, 3 hours.
PIECE_STEPS = {LAYOUT_ANALYSIS: 108, LAYOUT_FORECAST: 4, LAYOUT_ANALYSIS_PL: 3}


class S3BackendArray(BackendArray):
    """xarray ``BackendArray`` that reads windowed slabs from S3 on demand.

    Shape ``(n_time, n_lat, n_lon)`` or ``(n_time, n_level, n_lat, n_lon)``.
    Supports outer indexing (slices and 1-D integer arrays on each axis);
    xarray decomposes anything fancier onto that.
    """

    def __init__(self, index: VarIndex, window: Window, layout: str,
                 level_idx: tuple[int, ...] | None = None,
                 level_out_order: np.ndarray | None = None):
        self.index = index
        self.window = window
        self.layout = layout
        self.level_idx = level_idx
        self.level_out_order = level_out_order
        n_t = int(index.file_of_step.size)
        if level_idx is None:
            self.shape = (n_t, window.n_lat, window.n_lon)
        else:
            self.shape = (n_t, len(level_idx), window.n_lat, window.n_lon)
        self.dtype = np.dtype("float32")

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER, self._raw
        )

    def _raw(self, key: tuple) -> np.ndarray:
        n_t = self.shape[0]
        t_key = key[0]
        t_scalar = False
        if isinstance(t_key, slice):
            t_idx = np.arange(*t_key.indices(n_t))
        else:
            t_scalar = np.ndim(t_key) == 0          # an integer key drops the axis
            t_idx = np.atleast_1d(np.asarray(t_key, dtype="int64"))
            t_idx = np.where(t_idx < 0, t_idx + n_t, t_idx)
        out = np.full((t_idx.size, *self.shape[1:]), np.nan, dtype="float32")

        f_of = self.index.file_of_step[t_idx].astype("int64")
        l_of = self.index.local_of_step[t_idx].astype("int64")
        order = np.argsort(f_of * (2 ** 31) + l_of, kind="stable")
        f_s, l_s = f_of[order], l_of[order]

        # For forecast files a "stored step" is an initialisation holding 12
        # valid hours; convert flat valid-time indices to init indices.
        per_step = FC_HOURS.size if self.layout == LAYOUT_FORECAST else 1
        s_s = l_s // per_step
        piece = PIECE_STEPS.get(self.layout, 24)

        specs: list[ReadSpec] = []
        spans: list[tuple[int, int, int]] = []   # (i, j, lo_step) over the sorted order
        i = 0
        while i < t_idx.size:
            fi = int(f_s[i])
            if fi < 0:
                i += 1
                continue
            j = i
            while (j + 1 < t_idx.size and f_s[j + 1] == fi and s_s[j + 1] - s_s[i] < piece
                   and l_s[j + 1] == l_s[j] + 1):
                j += 1
            lo_step, hi_step = int(s_s[i]), int(s_s[j]) + 1
            specs.append(ReadSpec(self.index.files[fi], lo_step, hi_step, self.window,
                                  self.level_idx, self.index.factor))
            spans.append((i, j, lo_step))
            i = j + 1

        for (i, j, lo_step), block in zip(spans, _run(specs)):
            if self.layout == LAYOUT_FORECAST:
                block = block.reshape(-1, *block.shape[2:])          # (n_init*12, lat, lon)
            first = int(l_s[i]) - lo_step * per_step
            sel = block[first: first + (j - i + 1)]
            if self.level_out_order is not None:
                sel = sel[:, self.level_out_order]
            out[order[i: j + 1]] = sel

        # Spatial/level keys, applied one axis at a time (OUTER semantics --
        # numpy would broadcast two integer arrays) and from the LAST axis
        # backwards, so an integer key that drops an axis cannot shift the
        # axis numbers of the keys still to be applied.
        for axis in range(len(key) - 1, 0, -1):
            k = key[axis]
            if isinstance(k, slice):
                if k != slice(None):
                    out = out[(slice(None),) * axis + (k,)]
            elif np.ndim(k) == 0:
                out = out[(slice(None),) * axis + (int(k),)]
            else:
                out = np.take(out, np.asarray(k, dtype="int64"), axis=axis)
        return out[0] if t_scalar else out


# ----------------------------------------------------------------------------
# Time windows and the global axis
# ----------------------------------------------------------------------------
def _as_dt64h(x) -> np.datetime64:
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[h]")
    if isinstance(x, datetime):
        return np.datetime64(x.replace(minute=0, second=0, microsecond=0), "h")
    if isinstance(x, date):
        return np.datetime64(datetime(x.year, x.month, x.day), "h")
    s = str(x)
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d %H", "%Y-%m-%d"):
        try:
            return np.datetime64(datetime.strptime(s, fmt), "h")
        except ValueError:
            continue
    return np.datetime64(s, "h")


def _end_inclusive(end) -> np.datetime64:
    """A bare DATE as an end bound means 23:00 of that day, like ``--end``.

    Mirrors ``seb_analysis_common._inclusive_end``: a datetime at exactly
    00:00 (what ``parse_date("2026-01-07")`` returns) also means the whole day.
    """
    e = _as_dt64h(end)
    if isinstance(end, np.datetime64):
        bare = np.datetime_data(end)[0] in ("D", "M", "Y")
    elif isinstance(end, datetime):
        bare = (end.hour, end.minute) == (0, 0)
    elif isinstance(end, date):
        bare = True
    else:
        bare = ("T" not in str(end)) and (" " not in str(end))
    return e + np.timedelta64(23, "h") if bare else e


def normalise_windows(windows: Sequence[tuple], month_align: bool = True) -> list[tuple[np.datetime64, np.datetime64]]:
    """``[(start, end), ...]`` as hour-precision datetime64, end INCLUSIVE.

    With ``month_align`` (the default) each span is widened to whole calendar
    months, which is what the local archive gives ``load_seb_data(windows=)``
    too -- it opens whole files, and the season machinery then masks the time
    axis itself. Overlapping spans are merged so the axis is strictly increasing.
    """
    out = []
    for start, end in windows:
        s = _as_dt64h(start)
        e = _end_inclusive(end)
        if e < s:
            raise ValueError(f"window end {end} before start {start}")
        if month_align:
            s = s.astype("datetime64[M]").astype("datetime64[h]")
            e = (e.astype("datetime64[M]") + 1).astype("datetime64[h]") - np.timedelta64(1, "h")
        out.append((s, e))
    out.sort()
    merged: list[tuple[np.datetime64, np.datetime64]] = []
    for s, e in out:
        if merged and s <= merged[-1][1] + np.timedelta64(1, "h"):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def season_windows(years: Iterable[int], season_start: tuple[int, int],
                   season_end: tuple[int, int]) -> list[tuple[np.datetime64, np.datetime64]]:
    """One window per season START year, wrapping the new year when needed.

    Mirrors the season definition in the analysis modules: a season starting
    ``(10, 1)`` and ending ``(3, 31)`` for start year 2014 is
    2014-10-01 .. 2015-03-31 inclusive.
    """
    out = []
    for y in years:
        s = datetime(int(y), season_start[0], season_start[1])
        e_year = int(y) + 1 if tuple(season_end) < tuple(season_start) else int(y)
        e = datetime(e_year, season_end[0], season_end[1], 23)
        out.append((np.datetime64(s, "h"), np.datetime64(e, "h")))
    return out


def _global_axis(windows: list[tuple[np.datetime64, np.datetime64]]) -> np.ndarray:
    parts = []
    for s, e in windows:
        n = int((e - s).astype(int)) + 1
        parts.append(s + np.arange(n).astype("timedelta64[h]"))
    return np.concatenate(parts) if parts else np.empty(0, dtype="datetime64[h]")


def _months_of(axis: np.ndarray) -> list[tuple[int, int]]:
    months = np.unique(axis.astype("datetime64[M]"))
    return [(int(str(m)[:4]), int(str(m)[5:7])) for m in months]


def build_var_index(canonical: str, axis: np.ndarray, missing: str = "error") -> VarIndex:
    """Map every hour of ``axis`` to (file, flat index) for one variable."""
    source, factor = canonical, 1.0
    if canonical in DERIVED_VARS:
        source, factor = DERIVED_VARS[canonical][0], DERIVED_VARS[canonical][1]
    if source in UNAVAILABLE_ON_S3:
        raise KeyError(f"{canonical!r} is not in the NCAR bucket")
    if source not in NCAR_VARS:
        raise KeyError(f"no S3 mapping for {canonical!r}; add it to NCAR_VARS")

    files = files_for(source, _months_of(axis)) if axis.size else []
    file_of = np.full(axis.size, -1, dtype="int32")
    local_of = np.full(axis.size, -1, dtype="int32")
    for fi, f in enumerate(files):
        times = predicted_times(f)
        pos = np.searchsorted(axis, times)
        ok = pos < axis.size
        ok[ok] &= axis[pos[ok]] == times[ok]
        file_of[pos[ok]] = fi
        local_of[pos[ok]] = np.nonzero(ok)[0].astype("int32")
    n_missing = int((file_of < 0).sum())
    if n_missing and missing == "error":
        gaps = axis[file_of < 0]
        raise FileNotFoundError(
            f"{canonical!r}: {n_missing:,} of {axis.size:,} requested hours have "
            f"no file in s3://{BUCKET} (first {gaps[0]}, last {gaps[-1]}). The "
            f"bucket lags real time by 3-4 months (forecast groups one month more "
            f"than the analysis groups); set ERA5_S3_MISSING=nan or pass "
            f"missing='nan' to keep going with NaN there."
        )
    return VarIndex(canonical, files, file_of, local_of, factor, n_missing)


def _resolve_levels(levels_hpa: Sequence[float]):
    """Stored indices (ascending) + permutation into the requested order."""
    want = np.asarray(levels_hpa, dtype="float64")
    if np.unique(want).size != want.size:
        raise ValueError(f"levels_hpa has duplicates: {want}")
    pos = np.searchsorted(LEVELS_ALL_HPA, want)
    ok = (pos < LEVELS_ALL_HPA.size) & np.isclose(LEVELS_ALL_HPA[np.clip(pos, 0, 36)], want)
    if not ok.all():
        raise ValueError(f"levels not in the archive: {want[~ok]} (have {LEVELS_ALL_HPA})")
    stored_sorted = np.unique(pos)
    out_order = np.searchsorted(stored_sorted, pos)
    return want, tuple(int(i) for i in stored_sorted), out_order.astype("int64")


# ----------------------------------------------------------------------------
# Public constructors
# ----------------------------------------------------------------------------
def open_dataset(
    north_deg: float, west_deg: float, south_deg: float, east_deg: float,
    windows: Sequence[tuple],
    variables: Sequence[str] = SEB_STANDARD,
    levels_hpa: Sequence[float] | None = None,
    missing: str | None = None,
    month_align: bool = True,
    verbose: bool = True,
) -> xr.Dataset:
    """Lazy Dataset over a box and a set of time windows.

    Parameters
    ----------
    north_deg, west_deg, south_deg, east_deg
        CDS ``area`` order, longitudes on [-180, 180].
    windows
        ``[(start, end), ...]``, end inclusive; widened to whole months unless
        ``month_align=False``. Use :func:`season_windows` for cold seasons.
    variables
        Canonical names (surface and/or pressure level). ``tp`` is derived
        from ``mtpr``. Unknown or unavailable names raise before any I/O.
    levels_hpa
        Pressure levels to keep for pressure-level variables, in the order
        wanted in the output (the local archive uses descending, 1000 first).
        Required when any pressure-level variable is requested.
    missing
        ``"error"`` or ``"nan"`` for hours the bucket lacks; default from
        ``ERA5_S3_MISSING`` (``"error"``).

    Returns a Dataset with dims ``(valid_time, latitude, longitude)`` and, for
    pressure-level variables, ``(valid_time, pressure_level, latitude,
    longitude)``. Data variables are lazy until ``.load()``.
    """
    variables = list(dict.fromkeys(variables))
    missing = (missing or MISSING_DEFAULT).lower()
    if missing not in ("error", "nan"):
        raise ValueError(f"missing must be 'error' or 'nan', got {missing!r}")
    bad = [v for v in variables if v not in NCAR_VARS and v not in DERIVED_VARS]
    if bad:
        raise KeyError(f"no S3 mapping for {bad}"
                       + (f"; {sorted(set(bad) & UNAVAILABLE_ON_S3)} are not archived on S3"
                          if set(bad) & UNAVAILABLE_ON_S3 else "; add to NCAR_VARS"))

    window = make_window(north_deg, west_deg, south_deg, east_deg)
    wins = normalise_windows(windows, month_align=month_align)
    axis = _global_axis(wins)
    if axis.size == 0:
        raise ValueError("empty time window")

    pl_vars = [v for v in variables if v in PRESSURE_LEVEL_VARS]
    level_vals = level_idx = level_out = None
    if pl_vars:
        if levels_hpa is None:
            raise ValueError(f"levels_hpa is required for pressure-level variables {pl_vars}")
        level_vals, level_idx, level_out = _resolve_levels(levels_hpa)

    t_build = _time.time()
    data_vars = {}
    n_files = 0
    for name in variables:
        idx = build_var_index(name, axis, missing=missing)
        n_files += len(idx.files)
        spec = NCAR_VARS[DERIVED_VARS[name][0] if name in DERIVED_VARS else name]
        layout = GROUP_LAYOUT[spec.group]
        is_pl = name in PRESSURE_LEVEL_VARS
        arr = S3BackendArray(idx, window, layout,
                             level_idx if is_pl else None, level_out if is_pl else None)
        dims = (("valid_time", "pressure_level", "latitude", "longitude") if is_pl
                else ("valid_time", "latitude", "longitude"))
        if name in DERIVED_VARS:
            src, _, units, long_name, note = DERIVED_VARS[name]
            attrs = {"units": units, "long_name": long_name, "GRIB_paramId": 228,
                     "s3_note": note, "s3_source": src}
        else:
            attrs = {"units": spec.units, "long_name": spec.long_name,
                     "GRIB_paramId": idx.files[0].param_id if idx.files else -1}
        attrs["s3_group"] = spec.group
        attrs["s3_files"] = len(idx.files)
        if idx.n_missing:
            attrs["s3_missing_hours"] = idx.n_missing
        data_vars[name] = xr.Variable(dims, indexing.LazilyIndexedArray(arr), attrs=attrs)

    coords = {
        "valid_time": ("valid_time", axis.astype("datetime64[ns]"),
                       {"long_name": "time", "standard_name": "time"}),
        "latitude": ("latitude", window.lat_deg,
                     {"units": "degrees_north", "standard_name": "latitude", "long_name": "latitude",
                      "stored_direction": "decreasing"}),
        "longitude": ("longitude", window.lon_deg,
                      {"units": "degrees_east", "standard_name": "longitude", "long_name": "longitude"}),
    }
    if pl_vars:
        coords["pressure_level"] = ("pressure_level", level_vals,
                                    {"units": "hPa", "long_name": "pressure_level", "positive": "down"})

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update({
        "source": f"s3://{BUCKET} (NSF NCAR ERA5, anonymous), lazy via aws_pipeline.era5_s3",
        "convention": "ERA5: surface fluxes positive downward",
        "area_north_west_south_east_deg": f"[{north_deg}, {west_deg}, {south_deg}, {east_deg}]",
        "windows": [f"{s}..{e}" for s, e in wins],
    })
    if verbose:
        print(f"  S3 window : {window.n_lat} lat x {window.n_lon} lon "
              f"({north_deg}..{south_deg}N, {west_deg}..{east_deg}E)")
        print(f"  S3 time   : {axis.size:,} hourly steps in {len(wins)} window(s), "
              f"{axis[0]} .. {axis[-1]}")
        print(f"  S3 files  : {n_files} across {len(variables)} variables "
              f"(catalogued in {_time.time() - t_build:.1f} s, nothing read yet; "
              f"{n_workers()} reader threads; chunk cache "
              f"{'at ' + str(CACHE_DIR / 'chunks') if CHUNK_CACHE_ON else 'OFF'})")
    return ds


def open_land_sea_mask(north_deg: float, west_deg: float, south_deg: float,
                       east_deg: float) -> xr.DataArray:
    """The ERA5 invariant land-sea mask (land fraction, 0..1) for a box.

    The same field ``download_era5_land_sea_mask.py`` fetches from the CDS,
    read from ``e5.oper.invariant/197901``. Eager: it is one 1 MB file.
    """
    window = make_window(north_deg, west_deg, south_deg, east_deg)
    files = files_for("lsm", [])
    if not files:
        raise FileNotFoundError("no lsm file under e5.oper.invariant/197901")
    f = files[0]
    arr = read_window(ReadSpec(f, 0, 1, window, None, 1.0))
    da = xr.DataArray(
        arr.reshape(window.n_lat, window.n_lon),
        dims=("latitude", "longitude"),
        coords={"latitude": window.lat_deg, "longitude": window.lon_deg},
        name="lsm",
        attrs={"units": "(0 - 1)", "long_name": "Land-sea mask",
               "standard_name": "land_binary_mask", "GRIB_paramId": f.param_id,
               "GRIB_shortName": "lsm", "source": f"s3://{f.key}"},
    )
    return da


# ----------------------------------------------------------------------------
# Cost estimate
# ----------------------------------------------------------------------------
# Compressed bytes per unit (whole-globe chunks for the forecast and
# pressure-level groups regardless of window size); per analysis tile-day
# assumes a window covered by <=2 lon tiles.
BYTES_PER_UNIT = {LAYOUT_ANALYSIS: 1.0e6, LAYOUT_FORECAST: 18e6, LAYOUT_ANALYSIS_PL: 40e6}
# Laptop link measured 2026-09-18 with concurrent range requests: ~25 MB/s.
LAPTOP_MB_S = 25.0


def estimate_cost(windows: Sequence[tuple], variables: Sequence[str] = SEB_STANDARD,
                  month_align: bool = True) -> dict:
    """Rough transfer volume for a request, by variable and in total.

    The window size barely matters for the flux and pressure-level groups
    (whole-globe chunks), which is the main thing this is here to show.
    """
    wins = normalise_windows(windows, month_align=month_align)
    n_hours = sum(int((e - s).astype(int)) + 1 for s, e in wins)
    per_var, total_b = {}, 0.0
    for name in variables:
        src = DERIVED_VARS[name][0] if name in DERIVED_VARS else name
        layout = GROUP_LAYOUT[NCAR_VARS[src].group]
        units = {LAYOUT_ANALYSIS: n_hours / 24, LAYOUT_FORECAST: n_hours / 12,
                 LAYOUT_ANALYSIS_PL: n_hours}[layout]
        b = units * BYTES_PER_UNIT[layout]
        per_var[name] = {"layout": layout, "bytes": b}
        total_b += b
    return {"hours": n_hours, "per_variable": per_var, "GB": total_b / 1e9,
            "laptop_hours": total_b / 1e6 / LAPTOP_MB_S / 3600}


def describe_cost(windows: Sequence[tuple], variables: Sequence[str] = SEB_STANDARD,
                  month_align: bool = True) -> str:
    c = estimate_cost(windows, variables, month_align=month_align)
    by_layout: dict[str, float] = {}
    for v in c["per_variable"].values():
        by_layout[v["layout"]] = by_layout.get(v["layout"], 0.0) + v["bytes"]
    lines = [f"  {c['hours']:,} hourly steps x {len(variables)} variables: ~{c['GB']:.0f} GB "
             f"from S3 if every variable is read (~{c['laptop_hours']:.1f} h at "
             f"{LAPTOP_MB_S:.0f} MB/s; minutes in-region on EC2); a second pass is free "
             f"from the chunk cache"]
    for k, b in sorted(by_layout.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:<12} {b / 1e9:6.1f} GB")
    return "\n".join(lines)


__all__ = [
    "BUCKET", "NCAR_VARS", "DERIVED_VARS", "UNAVAILABLE_ON_S3", "SEB_STANDARD",
    "PRESSURE_LEVEL_VARS", "LEVELS_ALL_HPA", "CACHE_DIR",
    "open_dataset", "open_land_sea_mask", "season_windows", "normalise_windows",
    "make_window", "estimate_cost", "describe_cost", "list_month", "files_for",
    "predicted_times", "shutdown_pool", "n_workers",
]
