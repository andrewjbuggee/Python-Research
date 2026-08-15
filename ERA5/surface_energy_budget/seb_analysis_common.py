"""Shared loading, masking, and flux helpers for the ERA5 Arctic SEB analysis.

Used by ``plot_turbulent_flux_maps.py`` and ``plot_turbulent_flux_pdfs.py``.

SIGN CONVENTION
---------------
Everything in this module keeps the native **ERA5 convention: surface fluxes are
POSITIVE DOWNWARD (into the surface)**. That is the opposite of the Sledd et al.
(2025) Equation (1) convention implemented in ``seb_terms.py``, which flips the
turbulent terms to positive-upward. The two modules are deliberately different;
do not mix their outputs without checking the sign.

Under the ERA5 convention used here:

    net turbulent flux = msshf + mslhf        [W m-2, positive downward]

so a POSITIVE value means the turbulent fluxes are warming the surface, and a
NEGATIVE value means the surface is losing heat to the atmosphere.
"""

from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

# Storage roots are imported rather than redefined so the analysis side always
# reads from wherever the downloader writes. Editing EXTERNAL_ROOT in
# download_era5_seb.py moves both halves at once.
from download_era5_seb import STORAGE_ROOTS, check_volume_mounted

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

# Default location of the downloaded files, relative to this script. Same value
# as the downloader's LOCAL_ROOT.
DEFAULT_DATA_ROOT = STORAGE_ROOTS["local"]
DEFAULT_STORAGE = "local"

# Standard sea-ice-edge threshold. 15% is the long-standing convention for
# defining ice extent in passive-microwave sea ice climatology, and is what
# "ice free" normally means in the sea ice literature.
DEFAULT_MAX_SICONC = 0.15

# The three turbulent quantities plotted, in panel order.
FLUX_PANELS = (
    ("net_turbulent_W_m2", "Net turbulent flux", "SH + LH"),
    ("shf_W_m2", "Sensible heat flux", "msshf"),
    ("lhf_W_m2", "Latent heat flux", "mslhf"),
)

# The radiative counterparts, same 1x3 structure: the net first, then its
# components. ERA5 archives net radiative fluxes directly (msnlwrf, msnswrf),
# already positive downward, so no sign flip is involved here.
RADIATIVE_PANELS = (
    ("net_radiative_W_m2", "Net radiative flux", "LW$_{net}$ + SW$_{net}$"),
    ("lw_net_W_m2", "Net longwave flux", "msnlwrf"),
    ("sw_net_W_m2", "Net shortwave flux", "msnswrf"),
)


class MaskReport(NamedTuple):
    """Bookkeeping on how many cell-times survived masking."""

    n_total: int
    n_land: int
    n_ocean: int
    n_kept: int
    mask_mode: str
    max_siconc: float

    @property
    def pct_of_ocean(self) -> float:
        return 100.0 * self.n_kept / self.n_ocean if self.n_ocean else 0.0

    def describe(self) -> str:
        lines = [
            f"  total cell-times      : {self.n_total:,}",
            f"  land (siconc is NaN)  : {self.n_land:,} "
            f"({100 * self.n_land / self.n_total:.1f}%)",
            f"  ocean                 : {self.n_ocean:,} "
            f"({100 * self.n_ocean / self.n_total:.1f}%)",
        ]
        if self.mask_mode == "open-ocean":
            lines.append(
                f"  kept: ocean & siconc < {self.max_siconc:g} : {self.n_kept:,} "
                f"({self.pct_of_ocean:.3f}% of ocean)"
            )
        else:
            lines.append(
                f"  kept: all ocean incl. ice        : {self.n_kept:,} "
                f"({self.pct_of_ocean:.1f}% of ocean)"
            )
        return "\n".join(lines)


def parse_date(text: str) -> datetime:
    """Parse ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH`` into a datetime."""
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date {text!r}; use YYYY-MM-DD or YYYY-MM-DDTHH")


def available_regions(data_root: Path = DEFAULT_DATA_ROOT) -> list[str]:
    """Region subdirectories that actually contain netCDF files."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        return []
    return sorted(
        d.name for d in data_root.iterdir() if d.is_dir() and any(d.glob("*.nc"))
    )


# ----------------------------------------------------------------------------
# Where the data lives
# ----------------------------------------------------------------------------
def add_data_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the --storage / --data-root pair, identically across the scripts.

    Mirrors the downloader's own options so that whatever wrote the files can be
    named the same way when reading them back.
    """
    group = parser.add_argument_group("data source")
    group.add_argument(
        "--storage",
        choices=sorted(STORAGE_ROOTS),
        default=DEFAULT_STORAGE,
        help=(
            "Which disk to read from. 'local' is the data/ directory beside these "
            "scripts; 'external' is EXTERNAL_ROOT in download_era5_seb.py. These "
            "are the same two roots the downloader writes to. "
            f"(default: {DEFAULT_STORAGE})"
        ),
    )
    group.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="Explicit directory holding the region subdirectories, overriding --storage.",
    )
    group.add_argument(
        "--region",
        default="barrow",
        help=(
            "Region subdirectory to read. The downloader writes hourly data to "
            "<root>/<region>/ and other frequencies to <root>/<region>_<frequency>/, "
            "so 'barrow' and 'barrow_daily' are both valid. (default: barrow)"
        ),
    )


def resolve_data_root(storage: str = DEFAULT_STORAGE, data_root: Path | None = None) -> Path:
    """Turn ``--storage`` / ``--data-root`` into a concrete directory.

    Raises FileNotFoundError if an external root is requested while its volume is
    not mounted, which is a far clearer failure than an empty region listing.
    """
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    root = STORAGE_ROOTS[storage]
    if storage == "external":
        check_volume_mounted(root)
    return root


def resolve_region_dir(args: argparse.Namespace) -> Path:
    """Validate the requested region and return its directory.

    On a miss, checks the *other* storage location too, since asking the local
    disk for something that was downloaded to the external drive is the obvious
    mistake this option introduces.
    """
    data_root = resolve_data_root(args.storage, args.data_root)
    regions = available_regions(data_root)
    if args.region in regions:
        return data_root / args.region

    lines = [
        f"Region {args.region!r} not found under {data_root}",
        f"  available there: {regions or 'none'}",
    ]
    if args.data_root is None:
        other = "external" if args.storage == "local" else "local"
        try:
            other_root = resolve_data_root(other, None)
            other_regions = available_regions(other_root)
        except FileNotFoundError:
            other_regions = []
        if args.region in other_regions:
            lines.append(f"  but it IS on --storage {other} ({other_root})")
            lines.append(f"  rerun with: --storage {other}")
        elif other_regions:
            lines.append(f"  --storage {other} has: {other_regions}")
    raise FileNotFoundError("\n".join(lines))


def load_seb_data(
    region: str,
    start: datetime | None = None,
    end: datetime | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> "xr.Dataset":
    """Open every file for ``region`` and subset to the requested time range.

    The end bound is INCLUSIVE, so ``--end 2026-01-07`` keeps all 24 hours of
    7 January rather than stopping at 00:00.
    """
    import xarray as xr

    region_dir = Path(data_root) / region
    files = sorted(glob.glob(str(region_dir / "*.nc")))
    if not files:
        raise FileNotFoundError(
            f"No netCDF files in {region_dir}. "
            f"Available regions: {available_regions(Path(data_root)) or 'none'}"
        )

    # Two downloader runs with different --chunk-days over the same dates
    # produce differently-named files covering overlapping days. combine=
    # "by_coords" then fails with "does not have monotonic global indexes",
    # which is an accurate complaint but a fatal one. Fall back to an explicit
    # concat and drop duplicate timestamps: the overlapping values come from the
    # same archive and are identical, so keeping the first is lossless.
    try:
        # join and compat are passed explicitly rather than left to the
        # defaults. xarray warns that both defaults are changing (join
        # outer -> exact, compat no_conflicts -> override), and the new ones
        # would break this archive: files cover different time ranges, so an
        # exact join cannot align them. Naming the current values keeps the
        # behaviour pinned across an xarray upgrade instead of letting results
        # shift silently, and silences the warning as a side effect.
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            join="outer",           # union the per-file time ranges
            compat="no_conflicts",  # verify overlapping values agree
        )
    except ValueError as exc:
        if "monotonic" not in str(exc):
            raise
        # chunks={} keeps each file lazy; a plain open_dataset here would pull
        # every file fully into memory before the concat, which for a
        # multi-hundred-file archive is gigabytes to obtain a small subset.
        parts = [xr.open_dataset(f, chunks={}) for f in files]
        ds = xr.concat(parts, dim="valid_time", coords="minimal",
                       data_vars="minimal", compat="override", join="outer")
        ds = ds.sortby("valid_time")
        keep = ~ds.indexes["valid_time"].duplicated(keep="first")
        n_dupe = int((~keep).sum())
        ds = ds.isel(valid_time=keep)
        if n_dupe:
            print(
                f"  Note: {n_dupe:,} duplicate time steps dropped (files with "
                f"overlapping day ranges). Run with --overwrite off is unaffected; "
                f"see README on removing redundant files.",
                file=sys.stderr,
            )

    if start is not None or end is not None:
        # Push the end bound to the last instant of that day when no hour given.
        end_bound = end
        if end_bound is not None and (end_bound.hour, end_bound.minute) == (0, 0):
            end_bound = end_bound.replace(hour=23, minute=59, second=59)
        ds = ds.sel(valid_time=slice(start, end_bound))

    if ds.sizes.get("valid_time", 0) == 0:
        raise ValueError(
            f"No time steps in the requested range for region {region!r}. "
            f"Files span {_file_span(files)}."
        )
    return ds


def _file_span(files: list[str]) -> str:
    """Human-readable date span implied by the filenames."""
    stems = [Path(f).stem.split("_")[-1] for f in files]
    stems = [s for s in stems if len(s) == 8 and s.isdigit()]
    if not stems:
        return "an unknown range"
    return f"{min(stems)} to {max(stems)}"


def build_ocean_mask(
    ds: "xr.Dataset",
    mask_mode: str = "open-ocean",
    max_siconc: float = DEFAULT_MAX_SICONC,
) -> tuple["xr.DataArray", MaskReport]:
    """Boolean mask of the cell-times to analyse, plus a coverage report.

    ERA5 leaves ``siconc`` undefined (NaN) over land, so a finite value is the
    land/ocean test and its magnitude is the ice test. The NaN pattern was
    confirmed to be constant in time across the downloaded record, which is what
    you want from a land mask.

    Parameters
    ----------
    mask_mode :
        ``"open-ocean"`` keeps ocean cells with ``siconc < max_siconc``, i.e.
        ice-free water only. ``"all-ocean"`` keeps every ocean cell regardless of
        ice cover, dropping only land.
    max_siconc :
        Sea ice fraction below which a cell counts as open water. Ignored when
        ``mask_mode="all-ocean"``.
    """
    if "siconc" not in ds:
        raise KeyError(
            "Dataset has no 'siconc' (sea ice cover), so land and ice cannot be "
            "masked. Re-download with --var-set recommended or extended."
        )

    siconc = ds["siconc"]
    is_ocean = siconc.notnull()

    if mask_mode == "open-ocean":
        keep = is_ocean & (siconc < max_siconc)
    elif mask_mode == "all-ocean":
        keep = is_ocean
    else:
        raise ValueError(
            f"Unknown mask_mode {mask_mode!r}; use 'open-ocean' or 'all-ocean'."
        )

    n_total = int(siconc.size)
    n_ocean = int(is_ocean.sum())
    report = MaskReport(
        n_total=n_total,
        n_land=n_total - n_ocean,
        n_ocean=n_ocean,
        n_kept=int(keep.sum()),
        mask_mode=mask_mode,
        max_siconc=max_siconc,
    )
    return keep, report


def compute_turbulent_fluxes(
    ds: "xr.Dataset", mask: "xr.DataArray | None" = None
) -> "xr.Dataset":
    """Return the three turbulent flux fields in the ERA5 positive-downward sense.

    Returns a Dataset with ``net_turbulent_W_m2``, ``shf_W_m2`` and ``lhf_W_m2``.
    Where ``mask`` is False the values are NaN, so downstream means and histograms
    skip them automatically.
    """
    import xarray as xr

    missing = [v for v in ("msshf", "mslhf") if v not in ds]
    if missing:
        raise KeyError(
            f"Dataset is missing {missing}. Expected canonical ERA5 short names "
            f"(msshf, mslhf) as written by download_era5_seb.py."
        )

    shf_W_m2 = ds["msshf"]
    lhf_W_m2 = ds["mslhf"]

    if mask is not None:
        shf_W_m2 = shf_W_m2.where(mask)
        lhf_W_m2 = lhf_W_m2.where(mask)

    out = xr.Dataset(
        {
            "net_turbulent_W_m2": shf_W_m2 + lhf_W_m2,
            "shf_W_m2": shf_W_m2,
            "lhf_W_m2": lhf_W_m2,
        }
    )
    for name, _, _ in FLUX_PANELS:
        out[name].attrs["units"] = "W m-2"
    out["net_turbulent_W_m2"].attrs["long_name"] = "Net turbulent heat flux (SH + LH)"
    out["shf_W_m2"].attrs["long_name"] = "Surface sensible heat flux"
    out["lhf_W_m2"].attrs["long_name"] = "Surface latent heat flux"
    out.attrs["convention"] = "ERA5: positive downward (into the surface)"
    return out


def compute_radiative_fluxes(
    ds: "xr.Dataset", mask: "xr.DataArray | None" = None
) -> "xr.Dataset":
    """Return the three radiative flux fields in the ERA5 positive-downward sense.

    Returns a Dataset with ``net_radiative_W_m2``, ``lw_net_W_m2`` and
    ``sw_net_W_m2``, matching the RADIATIVE_PANELS order. ERA5's net radiative
    fluxes are already positive downward, so unlike the turbulent terms no sign
    convention change is applied.
    """
    import xarray as xr

    missing = [v for v in ("msnlwrf", "msnswrf") if v not in ds]
    if missing:
        raise KeyError(
            f"Dataset is missing {missing}. Expected canonical ERA5 short names "
            f"(msnlwrf, msnswrf) as written by download_era5_seb.py."
        )

    lw_net_W_m2 = ds["msnlwrf"]
    sw_net_W_m2 = ds["msnswrf"]
    if mask is not None:
        lw_net_W_m2 = lw_net_W_m2.where(mask)
        sw_net_W_m2 = sw_net_W_m2.where(mask)

    out = xr.Dataset(
        {
            "net_radiative_W_m2": lw_net_W_m2 + sw_net_W_m2,
            "lw_net_W_m2": lw_net_W_m2,
            "sw_net_W_m2": sw_net_W_m2,
        }
    )
    for name, title, _ in RADIATIVE_PANELS:
        out[name].attrs["units"] = "W m-2"
        out[name].attrs["long_name"] = title
    out.attrs["convention"] = "ERA5: positive downward (into the surface)"
    return out


def compute_net_seb(
    ds: "xr.Dataset", mask: "xr.DataArray | None" = None
) -> "xr.DataArray":
    """Net surface energy balance, positive downward (into the ocean).

        SEB_net = msnlwrf + msnswrf + msshf + mslhf   [W m-2]

    All four terms share ERA5's positive-downward convention, so the sum needs
    no sign flips. Positive means the surface is gaining energy; the sustained
    negative values of the freeze-up season are what cool the mixed layer and
    ultimately grow sea ice. Over OPEN ocean there is no conduction term to
    close: the residual goes into the water column.
    """
    missing = [v for v in ("msnlwrf", "msnswrf", "msshf", "mslhf") if v not in ds]
    if missing:
        raise KeyError(f"Dataset is missing {missing}; cannot form the net SEB.")
    net = ds["msnlwrf"] + ds["msnswrf"] + ds["msshf"] + ds["mslhf"]
    if mask is not None:
        net = net.where(mask)
    net.attrs["units"] = "W m-2"
    net.attrs["long_name"] = "Net surface energy balance (positive downward)"
    return net


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]
) -> list[float]:
    """Weighted quantiles of a 1-D sample, by inverting the weighted CDF.

    Same construction as the PDF script's stats box: sort, accumulate weights,
    interpolate at the requested quantile of total weight.
    """
    if values.size == 0:
        return [float("nan")] * len(qs)
    order = np.argsort(values)
    v_sorted = values[order]
    cum_w = np.cumsum(weights[order])
    cum_w = cum_w / cum_w[-1]
    return [float(np.interp(q, cum_w, v_sorted)) for q in qs]


# ----------------------------------------------------------------------------
# Season handling (shared; windows may wrap the new year)
# ----------------------------------------------------------------------------
def season_calendar(start_md, end_md) -> list[tuple[int, int]]:
    """Ordered (month, day) slots from start to end, wrapping the new year.

    A window like 09-01 to 03-31 spans two calendar years, so time is indexed by
    position within the season rather than by date. Built on a leap year so that
    29 February gets a slot; seasons without one simply leave that column empty.
    """
    import calendar as _cal
    from datetime import date as _date, timedelta as _td

    m0, d0 = start_md
    m1, d1 = end_md
    wraps = (m1, d1) < (m0, d0)
    # Whichever year holds February must be a leap year for 29 Feb to exist.
    if wraps:
        cur, stop = _date(1999, m0, d0), _date(2000, m1, d1)
    else:
        cur, stop = _date(2000, m0, d0), _date(2000, m1, d1)
    out = []
    while cur <= stop:
        out.append((cur.month, cur.day))
        cur += _td(days=1)
    return out


def season_year_of(times: np.ndarray, start_md) -> np.ndarray:
    """Calendar year each timestamp's season STARTS in.

    For a wrapping window, January belongs to the season that began the previous
    August, so those timestamps are labelled with year - 1.
    """
    years = times.astype("datetime64[Y]").astype(int) + 1970
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    m0, d0 = start_md
    before_start = (months * 100 + days) < (m0 * 100 + d0)
    return np.where(before_start, years - 1, years)


def season_slot_of(times: np.ndarray, slots: list[tuple[int, int]]) -> np.ndarray:
    """Index of each timestamp within the season, or -1 if outside the window.

    Uses a lookup on (month, day) rather than arithmetic on the date, which is
    what makes a wrapping window work: position in the season is defined by the
    slot list, not by how the calendar year happens to break.
    """
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
    lut = {md: i for i, md in enumerate(slots)}
    return np.array([lut.get((int(m), int(d)), -1) for m, d in zip(months, days)])


def hour_of(times: np.ndarray) -> np.ndarray:
    """Hour of day for each timestamp."""
    return (times.astype("datetime64[h]") - times.astype("datetime64[D]")).astype(int)


def area_weights(ds: "xr.Dataset") -> "xr.DataArray":
    """cos(latitude) weights, the standard area proxy on a regular lat/lon grid.

    Grid cell area scales as cos(lat), so a 70N cell covers roughly twice the area
    of an 80N cell at the same resolution. Without this, a pooled distribution
    over-represents the high-latitude end of the domain.
    """
    return np.cos(np.deg2rad(ds["latitude"]))


def warn_if_sparse(report: MaskReport, min_kept: int = 500) -> None:
    """Print a clear explanation when masking leaves too little to plot."""
    if report.n_kept >= min_kept:
        return
    print(
        f"\n  !! WARNING: only {report.n_kept:,} cell-times survived masking.",
        file=sys.stderr,
    )
    if report.mask_mode == "open-ocean":
        print(
            f"  !! Arctic winter sea ice covers nearly the whole domain, so the\n"
            f"  !! 'open-ocean' mask (siconc < {report.max_siconc:g}) removes almost\n"
            f"  !! everything. The plot will be mostly blank.\n"
            f"  !! Options: raise --max-siconc, use --mask all-ocean to keep\n"
            f"  !! ice-covered ocean, or pick a summer date range.",
            file=sys.stderr,
        )
