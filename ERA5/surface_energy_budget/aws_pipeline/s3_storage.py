"""``--storage aws``: the NCAR S3 bucket as a data root for every SEB analysis.

The analysis modules never open files themselves. Each ``prepare()`` does

    region_dir = resolve_region_dir(args)                      # seb_analysis_common
    ds         = load_seb_data(args.region, None, None, region_dir.parent)
    lsm_da     = load_land_sea_mask(args.region, resolve_data_root(...), args.mask_grid)

and streams ``ds`` through ``iter_time_blocks``. Those three functions carry a
one-line branch each: when the storage is ``"aws"`` they delegate here. So

    lwph.prepare(region="beaufort_chukchi", storage="aws",
                 years=range(2014, 2025), season_start=(10, 1), season_end=(3, 31), ...)

runs the identical code path as ``storage="local"``, reading from the bucket
instead of ``data/<region>/``. No analysis module is duplicated or edited.

What stands in for a directory
------------------------------
On disk a region is ``<root>/<region>/*.nc`` and the archive's time span is
whatever was downloaded. On S3 the span is 1940-2026, so "the whole archive"
is not a meaningful default: the time window must come from the caller.
:func:`resolve_region_dir` therefore builds a :class:`TimeSpec` from the same
``args`` every ``prepare()`` already has -- ``years`` + ``season_start`` +
``season_end`` (the season scripts) or ``start`` + ``end`` (the map/PDF
scripts) -- and hands back an :class:`S3RegionDir` whose ``.parent`` is an
:class:`S3DataRoot` carrying that spec. ``load_seb_data`` reads the spec off
the root. The lazy Dataset then covers exactly the requested hours -- the
season spans, or start..end -- which is what ``season_layout`` /
``select_seasons`` see on a locally downloaded season-only archive too.

Region names are the project's own (``era5_seb_variables.REGIONS``), a box
registered at run time with :func:`register_region`, or an inline
``"box:N,W,S,E"``. The ``_pressure`` / ``_pressure_wind`` suffixes select the
pressure-level archives the local downloader would have written next to a
region, with the same level sets (``era5_pressure_variables.LEVEL_SETS``).

The ARM-site caveat
-------------------
``plot_surface_class_timeseries.site_cell_mask`` picks the grid cell NEAREST
to Utqiagvik (71.323 N, 156.609 W) with no containment check. On a box that
does not contain it, every "ARM cell" quantity -- OV figures 1, 2, 3, 7a and
the site columns of 7b/7c -- silently describes an edge cell. This module
prints a prominent warning in that case; it does not raise, because the
domain-wide figures (4, 5, 6, 7b, 7c, 8) remain valid.
"""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # imported as aws_pipeline.s3_storage (the normal case)
    from . import era5_s3
except ImportError:  # aws_pipeline/ itself on sys.path
    import era5_s3  # type: ignore[no-redef]

AWS_STORAGE = "aws"

# ARM North Slope of Alaska site (Utqiagvik). Same numbers as
# plot_surface_class_timeseries.SITE_LAT / SITE_LON; duplicated here so this
# module never has to import the plotting stack, and checked against them at
# call time in _site_coords().
SITE_LAT_DEG = 71.323
SITE_LON_DEG = -156.609

# Suffixes the local layout uses for sibling archives of a region.
PRESSURE_SUFFIX = "_pressure"
WIND_SUFFIX = "_pressure_wind"
FREQ_SUFFIXES = ("_daily", "_monthly")

# What the local pressure-level archives hold (download_era5_pressure.py
# --var-set standard / winds, --levels troposphere / lower), reproduced from
# S3 with the same variables and level order (descending, 1000 hPa first).
PRESSURE_VARS: tuple[str, ...] = ("t", "q", "clwc", "ciwc", "cc")
WIND_VARS: tuple[str, ...] = ("u", "v")


# ----------------------------------------------------------------------------
# Region registry
# ----------------------------------------------------------------------------
_BOX_RE = re.compile(r"^box:\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*$")


def strip_suffixes(region: str) -> tuple[str, str]:
    """``("barrow", "_pressure_wind")`` for ``"barrow_pressure_wind"``."""
    for suf in (WIND_SUFFIX, PRESSURE_SUFFIX) + FREQ_SUFFIXES:
        if region.endswith(suf):
            return region[: -len(suf)], suf
    return region, ""


def register_region(name: str, north_deg: float, west_deg: float, south_deg: float,
                    east_deg: float, description: str = "") -> None:
    """Add a named box to ``era5_seb_variables.REGIONS`` for this session.

    After this, ``prepare(region=name, storage="aws")`` works, and so does
    the local downloader's ``--region name`` should the box ever be pulled to
    disk. Longitudes on [-180, 180], CDS ``[N, W, S, E]`` order.
    """
    import era5_seb_variables as v

    v.REGIONS[name] = v.Region(float(north_deg), float(west_deg), float(south_deg),
                               float(east_deg), description or f"registered box {name}")


def region_box(region: str) -> tuple[float, float, float, float]:
    """``(N, W, S, E)`` for a project region, a registered one, or ``box:N,W,S,E``."""
    base, _ = strip_suffixes(region)
    m = _BOX_RE.match(base)
    if m:
        return tuple(float(x) for x in m.groups())  # type: ignore[return-value]
    import era5_seb_variables as v

    r = v.get_region(base)
    return (r.north_deg, r.west_deg, r.south_deg, r.east_deg)


class S3RegionList(list):
    """The named regions, whose membership test also admits ``custom``,
    ``box:N,W,S,E`` and their ``_pressure`` / ``_pressure_wind`` siblings --
    anything :func:`region_box` can resolve -- since the bucket serves any box.
    (``turbulent_flux_response.cloud_temperature_field`` checks
    ``region_pl in available_regions(root)`` before loading.)"""

    def __contains__(self, name) -> bool:  # type: ignore[override]
        if list.__contains__(self, name):
            return True
        try:
            region_box(str(name))
            return True
        except (KeyError, ValueError):
            return False


def available_regions() -> S3RegionList:
    """Every region name the S3 root can serve (plus their PL siblings)."""
    import era5_seb_variables as v

    names = sorted(v.REGIONS)
    return S3RegionList(names + [n + PRESSURE_SUFFIX for n in names] + [n + WIND_SUFFIX for n in names])


# ----------------------------------------------------------------------------
# Time spec: the window the caller wants, derived from argparse Namespace
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeSpec:
    """Season windows or a plain start/end, as ``[(start, end), ...]``."""

    windows: tuple[tuple[np.datetime64, np.datetime64], ...]
    origin: str                          # how it was derived, for messages

    @classmethod
    def from_args(cls, args) -> "TimeSpec | None":
        years = getattr(args, "years", None)
        s0, s1 = getattr(args, "season_start", None), getattr(args, "season_end", None)
        if years is not None and s0 is not None and s1 is not None:
            # Exact season spans (e.g. 2014-10-01T00 .. 2015-03-31T23): the
            # season code masks by day-of-season anyway, and fetching only
            # these hours is what makes a partial-month season cheap.
            wins = era5_s3.normalise_windows(
                era5_s3.season_windows(list(years), tuple(s0), tuple(s1)), month_align=False)
            return cls(tuple(wins), f"years={tuple(years)} season {s0}..{s1}")
        start, end = getattr(args, "start", None), getattr(args, "end", None)
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("storage='aws' needs BOTH --start and --end (an open "
                                 "bound would mean the whole 1940-2026 archive)")
            # Exact, like load_seb_data's .sel(valid_time=slice(start, end)) on
            # disk; a bare-date end keeps that whole day (_inclusive_end).
            wins = era5_s3.normalise_windows([(start, end)], month_align=False)
            return cls(tuple(wins), f"start={start} end={end}")
        return None

    @classmethod
    def from_windows(cls, windows: Sequence[tuple]) -> "TimeSpec":
        return cls(tuple(era5_s3.normalise_windows(list(windows), month_align=True)),
                   "explicit windows")

    def describe(self) -> str:
        w = self.windows
        return (f"{len(w)} window(s) {w[0][0]}..{w[-1][1]} ({self.origin})"
                if w else "no windows")


# Fallback spec for callers that build a root through resolve_data_root()
# without going through resolve_region_dir(args) first -- e.g.
# turbulent_flux_response.cloud_temperature_field. Every prepare() calls
# resolve_region_dir(args) before anything else, which sets this.
_CURRENT_SPEC: TimeSpec | None = None


def current_spec() -> TimeSpec | None:
    return _CURRENT_SPEC


# ----------------------------------------------------------------------------
# Path-like stand-ins
# ----------------------------------------------------------------------------
class S3DataRoot:
    """What ``resolve_data_root("aws")`` returns: the bucket, plus a time spec.

    Supports the little the callers do with a data root: ``root / region``,
    ``str(root)``, ``is_dir()`` and ``exists()``.
    """

    def __init__(self, spec: TimeSpec | None = None):
        self._spec = spec

    @property
    def spec(self) -> TimeSpec | None:
        return self._spec if self._spec is not None else _CURRENT_SPEC

    def __truediv__(self, region: str) -> "S3RegionDir":
        return S3RegionDir(self, str(region))

    def __str__(self) -> str:
        return f"s3://{era5_s3.BUCKET}"

    def __repr__(self) -> str:
        return f"S3DataRoot({self.spec.describe() if self.spec else 'no time spec'})"

    def __fspath__(self) -> str:  # so Path(root) does not blow up in a print
        return str(self)

    def is_dir(self) -> bool:
        return True

    def exists(self) -> bool:
        return True

    def __eq__(self, other) -> bool:
        return isinstance(other, S3DataRoot)

    def __hash__(self) -> int:
        return hash("S3DataRoot")


class S3RegionDir:
    """What ``resolve_region_dir(args)`` returns for ``--storage aws``."""

    def __init__(self, root: S3DataRoot, region: str):
        self.root = root
        self.region = region
        self.box = region_box(region)  # validates the name early

    @property
    def parent(self) -> S3DataRoot:
        return self.root

    @property
    def name(self) -> str:
        return self.region

    def __truediv__(self, other: str) -> str:
        return f"{self}/{other}"

    def __str__(self) -> str:
        n, w, s, e = self.box
        spec = self.root.spec
        return (f"s3://{era5_s3.BUCKET} region={self.region} box=[{n}, {w}, {s}, {e}] "
                + (spec.describe() if spec else "(no time window yet)"))

    __repr__ = __str__

    def __fspath__(self) -> str:
        return str(self)

    def is_dir(self) -> bool:
        return True

    def exists(self) -> bool:
        return True


def is_s3_root(data_root) -> bool:
    return isinstance(data_root, (S3DataRoot, S3RegionDir))


def is_s3_storage(args) -> bool:
    return getattr(args, "storage", None) == AWS_STORAGE


# ----------------------------------------------------------------------------
# Hooks called from seb_analysis_common / surface_classification
# ----------------------------------------------------------------------------
def resolve_data_root() -> S3DataRoot:
    """``resolve_data_root("aws", ...)``: a root carrying the current time spec."""
    return S3DataRoot()


def resolve_region_dir(args) -> S3RegionDir:
    """``resolve_region_dir(args)`` for ``--storage aws``.

    Validates the region name and derives the time window from ``args``; the
    window travels on the returned object's ``.parent`` so the very next call,
    ``load_seb_data(args.region, None, None, region_dir.parent)``, knows what
    to fetch.
    """
    global _CURRENT_SPEC
    if getattr(args, "data_root", None) is not None:
        raise ValueError("--data-root has no meaning with --storage aws")
    spec = TimeSpec.from_args(args)
    if spec is None:
        raise ValueError(
            "storage='aws' needs a time window: pass years=(...) with "
            "season_start/season_end, or start/end. (On disk 'the whole archive' "
            "is whatever was downloaded; on S3 it would be 1940-2026.)")
    _CURRENT_SPEC = spec
    return S3RegionDir(S3DataRoot(spec), args.region)


def _windows_for(data_root, start, end, windows) -> list[tuple[np.datetime64, np.datetime64]]:
    """Precedence: explicit ``windows`` > ``start``/``end`` > the root's spec."""
    if windows:
        # Explicit windows (the local loader's own `windows=` argument): exact.
        return era5_s3.normalise_windows(list(windows), month_align=False)
    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("storage='aws' needs both start and end")
        # Explicit bounds are exact: the local loader subsets to them after
        # opening (end inclusive: a bare date keeps all 24 hours).
        return era5_s3.normalise_windows([(start, end)], month_align=False)
    root = data_root.parent if isinstance(data_root, S3RegionDir) else data_root
    spec = root.spec if isinstance(root, S3DataRoot) else None
    if spec is None or not spec.windows:
        raise ValueError(
            "storage='aws': no time window. Call resolve_region_dir(args) first "
            "(every prepare() does) or pass start/end or windows= to load_seb_data.")
    return list(spec.windows)


def load_seb_data(region: str, start=None, end=None, data_root=None, windows=None,
                  verbose: bool = True):
    """``load_seb_data`` for an S3 root: the lazy Dataset for ``region``.

    ``<name>_pressure`` and ``<name>_pressure_wind`` give the pressure-level
    datasets (dims ``valid_time, pressure_level, latitude, longitude``) with
    the same variables and level order as the local downloader writes;
    anything else gives the single-level SEB set.
    """
    import era5_pressure_variables as pv

    base, suffix = strip_suffixes(region)
    n, w, s, e = region_box(region)
    wins = _windows_for(data_root, start, end, windows)
    if verbose:
        print(f"  S3 region : {region} -> box [{n}, {w}, {s}, {e}], {len(wins)} window(s)")

    if suffix == PRESSURE_SUFFIX:
        levels = [float(p) for p in pv.LEVEL_SETS["troposphere"]]
        ds = era5_s3.open_dataset(n, w, s, e, wins, variables=PRESSURE_VARS,
                                  levels_hpa=levels, month_align=False, verbose=verbose)
    elif suffix == WIND_SUFFIX:
        levels = [float(p) for p in pv.LEVEL_SETS["lower"]]
        ds = era5_s3.open_dataset(n, w, s, e, wins, variables=WIND_VARS,
                                  levels_hpa=levels, month_align=False, verbose=verbose)
    else:
        # `wins` are already normalised by _windows_for -- do not let
        # open_dataset re-align them. Hours the bucket lacks raise unless
        # ERA5_S3_MISSING=nan (era5_s3.MISSING_DEFAULT).
        ds = era5_s3.open_dataset(n, w, s, e, wins, variables=era5_s3.SEB_STANDARD,
                                  month_align=False, verbose=verbose)
        if verbose:
            _warn_if_site_outside(ds, region)
            print(era5_s3.describe_cost(wins, era5_s3.SEB_STANDARD, month_align=False))
    return ds


def region_time_index(region: str, data_root) -> np.ndarray:
    """``region_time_index`` for an S3 root: the hourly axis of the time spec."""
    wins = _windows_for(data_root, None, None, None)
    axis = era5_s3._global_axis(wins)
    return axis.astype("datetime64[ns]")


def load_land_sea_mask(region: str, grid_deg: float | None = None):
    """``load_land_sea_mask`` for an S3 root: the invariant lsm, native grid only."""
    if grid_deg is not None and not np.isclose(grid_deg, era5_s3.GRID_DEG):
        raise ValueError(f"storage='aws' serves the land-sea mask on the native "
                         f"{era5_s3.GRID_DEG} deg grid only; got --mask-grid {grid_deg}")
    n, w, s, e = region_box(region)
    return era5_s3.open_land_sea_mask(n, w, s, e)


# ----------------------------------------------------------------------------
# Pressure-level datasets for cloud_level_wind
# ----------------------------------------------------------------------------
def pressure_level_datasets(E, pressure_suffix: str = PRESSURE_SUFFIX,
                            wind_suffix: str = WIND_SUFFIX, require_wind: bool = False):
    """``(ds_c, ds_w)`` for ``cloud_level_wind.cloud_level_wind`` under S3.

    Replaces the directory scan + ``open_mfdataset`` with lazy S3 datasets
    over the run's own hours (``E.times``), clwc on the troposphere levels and
    u, v on the lower-troposphere levels, exactly as the local archives hold
    them. ``require_wind`` is moot: the bucket has u and v for every month.
    """
    t = np.asarray(E.times).astype("datetime64[h]")
    # Whole months touched by the run: the same granularity as the local files.
    months = np.unique(t.astype("datetime64[M]"))
    windows = [(str(m), str(m + 1)) for m in months]   # [month start, next month start)
    wins = [(np.datetime64(a, "h"), np.datetime64(b, "h") - np.timedelta64(1, "h"))
            for a, b in windows]
    region = E.args.region
    ds_c = load_seb_data(region + pressure_suffix, windows=wins, data_root=S3DataRoot(),
                         verbose=False)[["clwc"]]
    ds_w = load_seb_data(region + wind_suffix, windows=wins, data_root=S3DataRoot(),
                         verbose=False)[["u", "v"]]
    print(f"  S3 pressure levels: clwc on {ds_c.sizes['pressure_level']} levels, "
          f"u/v on {ds_w.sizes['pressure_level']} levels, {ds_c.sizes['valid_time']:,} hours")
    return ds_c, ds_w


# ----------------------------------------------------------------------------
# ARM-site containment warning
# ----------------------------------------------------------------------------
def _site_coords() -> tuple[float, float]:
    try:
        from plot_surface_class_timeseries import SITE_LAT, SITE_LON  # type: ignore

        if not (np.isclose(SITE_LAT, SITE_LAT_DEG) and np.isclose(SITE_LON, SITE_LON_DEG)):
            warnings.warn(f"SITE_LAT/LON in plot_surface_class_timeseries "
                          f"({SITE_LAT}, {SITE_LON}) differ from s3_storage's copy", stacklevel=2)
        return float(SITE_LAT), float(SITE_LON)
    except Exception:  # noqa: BLE001 - plotting stack unavailable; use the copy
        return SITE_LAT_DEG, SITE_LON_DEG


def site_in_domain(ds) -> bool:
    """Does the dataset's box contain the ARM Utqiagvik cell (within half a cell)?"""
    lat, lon = _site_coords()
    la = np.asarray(ds["latitude"].values, dtype=float)
    lo = np.asarray(ds["longitude"].values, dtype=float)
    dlat = np.min(np.abs(la - lat))
    dlon = np.min(np.abs(((lo - lon) + 180.0) % 360.0 - 180.0))
    return bool(dlat <= era5_s3.GRID_DEG / 2 + 1e-6 and dlon <= era5_s3.GRID_DEG / 2 + 1e-6)


def _warn_if_site_outside(ds, region: str) -> None:
    if site_in_domain(ds):
        return
    lat, lon = _site_coords()
    msg = (
        f"\n  !! Region {region!r} does NOT contain the ARM Utqiagvik cell "
        f"({lat:.3f} N, {lon:.3f} E).\n"
        f"  !! site_cell_mask() will silently pick the nearest EDGE cell, so every\n"
        f"  !! 'ARM cell' quantity -- OV figures 1, 2, 3, 7a and the site columns of\n"
        f"  !! 7b/7c -- describes that edge cell, not Utqiagvik. Domain-wide figures\n"
        f"  !! (4, 5, 6, 7b, 7c, 8) are unaffected.\n"
    )
    print(msg, file=sys.stderr, flush=True)
    warnings.warn(f"region {region!r} does not contain the ARM site", stacklevel=3)


__all__ = [
    "AWS_STORAGE", "S3DataRoot", "S3RegionDir", "TimeSpec", "is_s3_root", "is_s3_storage",
    "register_region", "region_box", "available_regions", "resolve_data_root",
    "resolve_region_dir", "load_seb_data", "region_time_index", "load_land_sea_mask",
    "pressure_level_datasets", "site_in_domain", "strip_suffixes",
]
