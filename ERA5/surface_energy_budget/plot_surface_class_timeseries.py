#!/usr/bin/env python3
"""Seasonal time series of a variable averaged within five surface classes.

Two stacked panels, sharing a day-of-season x axis that runs 08-01 to 03-31 by
default:

    top     the variable of interest, one line per surface class
    bottom  the % of the domain's area occupied by each class, same colours

The classification (land / coastal / open ocean / marginal ice zone / sea ice)
is done per grid cell per time step by ``surface_classification.py``; see that
module for the thresholds and for why land is detected from ``lsm`` rather than
from ``siconc == 0``.

Why the bottom panel is not decoration
======================================
The ice edge moves several degrees between years, so a class's line in the top
panel is a mean over a footprint that is itself changing size. Late in the
season "open ocean" may be 1% of the domain, and its mean is then a noisy
average over a handful of cells that says little about the region. The bottom
panel is what tells you when to believe the top one. For the same reason a
class's line is BLANKED wherever it holds less than ``--min-class-area`` percent
of the domain, rather than drawn as a spike.

Quantities (``--variable``, default ``all``, one figure each)
============================================================
    dlr               downwelling longwave, msdwlwrf              [W m-2]
    lwp               liquid water path, tclw                     [g m-2]
    iwp               ice water path, tciw                        [g m-2]
    liquid_fraction   cloudy scenes containing liquid             [%]

``lwp`` and ``iwp`` zero any hour at or below ``--trace-threshold`` before
averaging, the same guard against ERA5's near-zero trace condensate that
``plot_monthly_lwp_maps.py`` applies before its median.

``liquid_fraction`` uses the definition from
``plot_latitude_time_hovmoller_liquid_fraction.py``::

    cloudy = tcc >= --min-cloud-fraction AND (LWP > --lwp-threshold
                                              OR IWP > --iwp-threshold)
    liquid = cloudy AND LWP > --lwp-threshold
    fraction = area-weighted cloudy cell-hours with liquid / cloudy cell-hours

Three thresholds, one unit, two different jobs
===============================================
``--lwp-threshold``, ``--iwp-threshold`` and ``--trace-threshold`` are ALL in
g m-2, so nothing on this script's command line needs converting in your head
-- but they are not interchangeable:

    --lwp-threshold (default 5)      decides whether liquid is PRESENT, for
                                      the liquid_fraction panel's cloudy/
                                      liquid tests.
    --iwp-threshold (default 0)      the same PRESENT test, for ice, used only
                                      in the liquid_fraction panel's cloudy
                                      test.
    --trace-threshold (default 0.03) unrelated to "present" at all: it only
                                      zeroes near-zero LWP/IWP hours before the
                                      lwp/iwp panels average a MAGNITUDE, so
                                      ERA5's numerical floor does not inflate
                                      an otherwise-dry period.

How the reduction works
=======================
Means are area-weighted by cos(latitude) and pooled over every cell-hour in a
day-of-season slot, then averaged across seasons with equal weight per season.
Pooling rather than averaging per-hour means keeps a slot with partial coverage
honest.

The mean, not the median, is used: it composes correctly across the streaming
blocks this script reads in, which a median does not. For LWP that matters --
the distribution is strongly skewed, so these means sit above the medians that
``plot_monthly_lwp_maps.py`` reports. The two are not directly comparable.

OPTIONS
=======

Data source
-----------
--storage {local,external}   Which disk to read (default local).
--data-root PATH             Explicit directory, overriding --storage.
--region NAME                Region subdirectory (default barrow).

Surface classification (see surface_classification.py for the logic itself)
-----------------------------------------------------------------------------
--lsm-tol TOL                 How close lsm must be to 0 or 1 to count as
                              pure land/sea; packing round-off insurance, not
                              physics (default 1e-4).
--open-ocean-max-siconc F     Sea ice below this is open ocean (default 0.05).
--sea-ice-min-siconc F        Sea ice above this is pack ice (default 0.95).
                              Between the two cuts is the marginal ice zone.
--land-max-siconc F           Ice cover a pure-land cell may still carry and
                              count as land (default 0.001).
--mask-grid DEG               Read the land-sea mask at this regridded
                              resolution instead of native.

Variable and season
--------------------
--variable {dlr,lwp,iwp,liquid_fraction,all} [...]
                              Which quantity/quantities to plot, one figure
                              each (default: all four). Repeatable.
--season-start MM-DD          First day of the window (default 08-01).
--season-end MM-DD            Last day, inclusive (default 03-31). The window
                              MAY wrap the new year, which this default does.
--years SPEC                  Seasons to average, by the year each season
                              STARTS in: '2019', '2019-2025', or
                              '2000,2019-2020'. Default: every season meeting
                              --min-season-coverage.
--min-season-coverage F       Exclude seasons covering less than this
                              fraction of the window from a climatology
                              (default 0.6).

Thresholds -- all in g m-2 (see "Three thresholds, one unit" above)
-----------------------------------------------------------------------
--lwp-threshold G              LWP above which liquid counts as PRESENT, used
                               by the liquid_fraction panel's cloudy and
                               liquid tests (default 5).
--iwp-threshold G              IWP above which it counts toward the
                               liquid_fraction panel's cloudy test
                               (default 0).
--min-cloud-fraction F         Total cloud cover at or above which a scene
                               counts as cloudy, for the liquid_fraction
                               panel (default 1, fully overcast).
--trace-threshold G            LWP/IWP at or below which an hour is treated
                               as zero before the lwp/iwp MAGNITUDE mean
                               (default 0.03). Independent of
                               --lwp-threshold/--iwp-threshold.

Surface-class line and area display
--------------------------------------
--min-class-area PCT          Blank a class's line in the top panel wherever
                              it holds less than this %% of the domain area
                              (default 0.5). Set 0 to draw everything.
--smooth DAYS                 Centred, NaN-aware running mean applied to the
                              TOP panel only, in days of season (default 1,
                              i.e. none).
--area-style {stacked,lines}  Bottom panel as a stacked area (default) or as
                              one line per class.

Performance
-----------
--block-hours N                Time steps held in memory at once while
                               streaming the archive (default 720, i.e. 30
                               days of hourly data).

Output
------
--output-dir PATH              Where figures go (default ./figures).
--dpi N                        Figure resolution (default 200).
--show                         Open interactive windows.

Examples
========
  # all four figures, every season with adequate coverage
  ./plot_surface_class_timeseries.py --region barrow

  # one season, one variable
  ./plot_surface_class_timeseries.py --region barrow --years 2021 --variable dlr

  # a 7-season climatology of the liquid fraction
  ./plot_surface_class_timeseries.py --region barrow --years 2019-2025 \
      --variable liquid_fraction
"""

from __future__ import annotations

import argparse
import calendar
import sys
import warnings
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seb_analysis_common import (
    add_data_source_args,
    format_used_seasons,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
    season_calendar,
    season_slot_of,
    season_year_of,
)
from surface_classification import (
    CLASS_CODES,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    UNCLASSIFIED,
    DEFAULT_BLOCK_HOURS,
    add_classification_args,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

# quantity -> (label, units, kind). "mean" quantities average a field; "ratio"
# quantities divide two weighted counts.
QUANTITIES: dict[str, tuple[str, str, str]] = {
    "dlr": ("Downwelling longwave", "W m$^{-2}$", "mean"),
    "lwp": ("Liquid water path", "g m$^{-2}$", "mean"),
    "iwp": ("Ice water path", "g m$^{-2}$", "mean"),
    "liquid_fraction": ("Cloudy scenes with liquid", "%", "ratio"),
}

# ERA5 source variable and unit scaling for each "mean" quantity.
MEAN_SOURCES: dict[str, tuple[str, float]] = {
    "dlr": ("msdwlwrf", 1.0),
    "lwp": ("tclw", 1000.0),   # kg m-2 -> g m-2
    "iwp": ("tciw", 1000.0),   # kg m-2 -> g m-2
}

# Matches plot_latitude_time_hovmoller_liquid_fraction.py's and
# analyze_cloud_liquid_frequency.py's default values. All three scripts (and
# the trace threshold below) share one unit, g m-2, so nothing on any of their
# command lines needs a unit conversion in your head.
DEFAULT_LWP_THRESHOLD_G = 5.0
DEFAULT_IWP_THRESHOLD_G = 0.0
DEFAULT_MIN_CLOUD_FRACTION = 1.0

# Matches plot_monthly_lwp_maps.py's trace guard.
# MEASURED: ERA5's tclw/tciw here are quantised to exact multiples of
# 2**-15 kg m-2 = 0.0305176 g m-2 (a GRIB binary scale factor), so the smallest
# non-zero path the archive can express is 0.03052 g m-2. A threshold BELOW that
# quantum selects exactly the same cell-hours as "> 0" and guards nothing. To
# actually drop the single-quantum population, use >= 0.031.
DEFAULT_TRACE_THRESHOLD_G = 0.03

# A class holding less than this share of the domain has its line blanked.
DEFAULT_MIN_CLASS_AREA_PCT = 0.5

REQUIRED_VARS = ("msdwlwrf", "tclw", "tciw", "tcc", "siconc")

# ----------------------------------------------------------------------------
# The DOE ARM site at Utqiagvik, as a SIXTH series that is NOT a sixth class.
# ----------------------------------------------------------------------------
# The single grid cell containing the ARM facility, carried alongside the five
# surface classes so ERA5 can be compared against the ground observations there.
#
# It is deliberately NOT a member of CLASS_ORDER. The five classes partition the
# domain and their area shares sum to 100%; this cell is already counted inside
# whichever class it falls in (coastal, in this region), so adding it to that
# partition would double-count it and push the stacked area panel past 100%.
# It is therefore accumulated on its own axis slot, drawn as an overlay on the
# top panel, and kept out of the area panel entirely. Its own area share is
# about 1/N_cells -- 0.04% of the Barrow strip -- which is meaningless as a
# share and is reported only as a sanity check that exactly one cell was picked.
SITE_KEY = "arm_site"
SITE_LABEL = "Utqiagvik (ARM site)"
SITE_COLOR = "#111111"
SITE_LAT = 71.323      # deg N, DOE ARM North Slope of Alaska central facility
SITE_LON = -156.609    # deg E (negative = W)


def site_cell_mask(ds, lat_deg=SITE_LAT, lon_deg=SITE_LON):
    """Boolean (lat, lon) mask with exactly one True: the cell holding the site.

    Nearest-neighbour on the ERA5 grid. Longitudes are compared on a signed
    -180..180 axis so a 0..360 archive matches too. Returns the mask and the
    (lat, lon) of the chosen cell centre so the caller can report how far the
    cell centre sits from the facility.
    """
    lats = np.asarray(ds["latitude"].values, dtype=float)
    lons = np.asarray(ds["longitude"].values, dtype=float)
    lons_signed = ((lons + 180.0) % 360.0) - 180.0
    i = int(np.argmin(np.abs(lats - lat_deg)))
    j = int(np.argmin(np.abs(((lons_signed - lon_deg) + 180.0) % 360.0 - 180.0)))
    mask = np.zeros((lats.size, lons.size), dtype=bool)
    mask[i, j] = True
    return mask, float(lats[i]), float(lons_signed[j])


def nanmean_quiet(a, axis=None):
    """np.nanmean without the "Mean of empty slice" RuntimeWarning.

    All-NaN slices are expected here, not a mistake: a class can be absent from
    the domain for a whole slot, and a season may not cover the full window.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def parse_years(text: str) -> tuple[int, ...]:
    """Parse '2019', '2019-2025', or '2000,2019-2020' into a sorted year tuple."""
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part.lstrip("-"):
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if hi < lo:
                    raise ValueError
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{part!r} is not a year or an ascending YYYY-YYYY range"
            ) from None
    if not out:
        raise argparse.ArgumentTypeError("no years given")
    return tuple(sorted(out))


def parse_month_day(text: str) -> tuple[int, int]:
    try:
        mm, dd = text.split("-")
        month, day = int(mm), int(dd)
        date(2000, month, day)  # 2000 is a leap year, so 02-29 validates
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid MM-DD") from None
    return month, day


def running_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centred, NaN-aware running mean over ``window`` day-of-season slots.

    NaN-aware rather than a plain convolution because a class's line is blanked
    wherever it is too small to trust; a normal moving average would drag those
    gaps outward and eat real data on either side of them.
    """
    if window <= 1:
        return y
    kernel = np.ones(window)
    finite = np.isfinite(y)
    num = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def month_ticks(slots: list[tuple[int, int]]) -> tuple[list[int], list[str]]:
    """Tick at the first of each month present, labelled by month abbreviation."""
    pos, lab = [], []
    for i, (m, d) in enumerate(slots):
        if d == 1:
            pos.append(i)
            lab.append(calendar.month_abbr[m])
    return pos, lab


# ----------------------------------------------------------------------------
# Reduction
# ----------------------------------------------------------------------------
def season_layout(ds, args) -> dict:
    """Map every time step to a season and a day-of-season slot.

    Derived from the time coordinate ALONE, so it is cheap and, critically,
    available before any field is read. Season selection therefore happens
    first and only the chosen seasons are ever loaded.

    ``ds`` may be a Dataset or a bare array of timestamps. The array form is
    what lets the caller run this against ``region_time_index()`` -- the cached
    file index -- and so choose its seasons before opening a single file.
    """
    times = ds["valid_time"].values if hasattr(ds, "data_vars") else np.asarray(ds)
    slots = season_calendar(args.season_start, args.season_end)
    dos = season_slot_of(times, slots)
    seasons = season_year_of(times, args.season_start)

    in_window = dos >= 0
    if not in_window.any():
        raise ValueError("No time steps fall inside the requested season window.")
    uniq_seasons = sorted(set(seasons[in_window].tolist()))

    # Coverage of the window per season, again from the time axis only.
    counts = np.zeros((len(uniq_seasons), len(slots)), dtype=int)
    season_index = {s: i for i, s in enumerate(uniq_seasons)}
    s_idx = np.array([season_index.get(int(s), -1) for s in seasons])
    sel = in_window & (s_idx >= 0)
    np.add.at(counts, (s_idx[sel], dos[sel]), 1)

    return {
        "slots": slots, "dos": dos, "s_idx": s_idx,
        "in_window": in_window, "seasons": uniq_seasons, "counts": counts,
    }


def select_seasons(layout: dict, args) -> tuple[list[int], list[int], str]:
    """Pick which seasons to use, printing the coverage table as it goes.

    Shared with the bar-chart script so the two never disagree about which
    seasons a given set of options selects. Raises ValueError with a usable
    message rather than exiting, so the caller controls its own error path.

    Returns ``(indices into layout["seasons"], the season years, a label)``.
    """
    n_slot = len(layout["slots"])
    print(f"\n  Seasons found ({len(layout['seasons'])}), coverage of the "
          f"{n_slot}-day window:")
    frac = {}
    for s_i, s in enumerate(layout["seasons"]):
        f = float((layout["counts"][s_i] > 0).sum()) / n_slot
        frac[s] = f
        print(f"    {s}/{s+1}: {f*100:5.1f}%"
              + ("" if f >= args.min_season_coverage else "   (below --min-season-coverage)"))

    if args.years is not None:
        absent = [y for y in args.years if y not in layout["seasons"]]
        if absent:
            raise ValueError(f"no data for season(s) {absent}. "
                             f"Available: {layout['seasons']}")
        # An explicit request is honoured as given: --min-season-coverage only
        # guards the automatic default.
        keep_idx = [layout["seasons"].index(y) for y in args.years]
        thin = [y for y in args.years if frac[y] < args.min_season_coverage]
        if thin:
            print(f"\n  !! Requested season(s) {thin} cover less than "
                  f"{args.min_season_coverage:.0%} of the window and are included "
                  f"anyway because you named them.", file=sys.stderr)
    else:
        keep_idx = [i for i, s in enumerate(layout["seasons"])
                    if frac[s] >= args.min_season_coverage]
        if not keep_idx:
            raise ValueError(f"no season meets --min-season-coverage "
                             f"{args.min_season_coverage}.")

    used = [layout["seasons"][i] for i in keep_idx]
    label = format_used_seasons(used)
    return keep_idx, used, label


def build_series(ds, lsm: np.ndarray, args, layout: dict,
                 wanted_idx: list[int]) -> dict:
    """Reduce the chosen seasons to (season, day-of-season, class) arrays.

    Every quantity is accumulated in ONE streaming pass, because the four
    figures share the same classification and the same masking; reading the
    archive four times to produce them would be the dominant cost.

    Only seasons in ``wanted_idx`` are read. Their slots keep their position in
    the full season list so the caller's indexing stays valid.
    """
    slots = layout["slots"]
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    uniq_seasons = layout["seasons"]

    wanted = np.zeros(len(uniq_seasons), dtype=bool)
    wanted[wanted_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    # One extra slot on the class axis for the ARM site cell. It rides along
    # through every accumulator so the site gets the same treatment as a class,
    # without being one -- see SITE_KEY above.
    site_mask, site_lat, site_lon = site_cell_mask(ds)
    site_code = len(CLASS_ORDER)

    n_season, n_slot = len(uniq_seasons), len(slots)
    n_class = len(CLASS_ORDER) + 1
    shape = (n_season, n_slot, n_class)

    w_class = np.zeros(shape)          # area weight per class
    w_domain = np.zeros((n_season, n_slot))   # total domain weight, all cells
    vw = {q: np.zeros(shape) for q in MEAN_SOURCES}       # sum(w * value)
    vw_w = {q: np.zeros(shape) for q in MEAN_SOURCES}     # sum(w) over finite values
    cloudy_w = np.zeros(shape)
    liquid_w = np.zeros(shape)

    lat_deg = ds["latitude"].values
    weights = area_weights_2d(lat_deg, ds.sizes["longitude"])
    w_domain_per_step = float(weights.sum())

    n_unclassified = 0
    # Which of the five classes the site cell actually falls in. Recorded per
    # time step rather than assumed, because a coastal cell's class can move
    # with the sea ice; the report prints the breakdown.
    site_class_counts = np.zeros(len(CLASS_ORDER) + 1, dtype=np.int64)

    for i0, block in iter_time_blocks(ds, list(REQUIRED_VARS), args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        sl = slice(i0, i0 + n_t)
        keep = use_step[sl]

        siconc = block["siconc"].values
        classes = classify_cells(
            lsm, siconc, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )
        n_unclassified += int((classes == -1).sum())

        site_codes = classes[:, site_mask][keep]      # (n_kept, 1)
        for code in range(len(CLASS_ORDER)):
            site_class_counts[code] += int((site_codes == code).sum())
        site_class_counts[-1] += int((site_codes == UNCLASSIFIED).sum())

        tclw_g = block["tclw"].values * 1000.0
        tciw_g = block["tciw"].values * 1000.0
        tcc = block["tcc"].values

        # Trace guard for the magnitude quantities only; the liquid test below
        # uses its own, much higher, threshold.
        values = {
            "dlr": block["msdwlwrf"].values,
            "lwp": np.where(tclw_g > args.trace_threshold, tclw_g, 0.0),
            "iwp": np.where(tciw_g > args.trace_threshold, tciw_g, 0.0),
        }

        valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw_g)
        cloudy = valid & (tcc >= args.min_cloud_fraction) & (
            (tclw_g > args.lwp_threshold)
            | (tciw_g > args.iwp_threshold)
        )
        liquid = cloudy & (tclw_g > args.lwp_threshold)

        si = s_idx[sl][keep]
        di = dos[sl][keep]
        np.add.at(w_domain, (si, di), w_domain_per_step)

        selectors = [(CLASS_CODES[n], classes == CLASS_CODES[n])
                     for n in CLASS_ORDER]
        selectors.append((site_code, np.broadcast_to(site_mask, classes.shape)))
        for code, sel in selectors:
            wc = sel * weights                        # (n_t, lat, lon)
            np.add.at(w_class, (si, di, code), wc.sum(axis=(1, 2))[keep])
            np.add.at(cloudy_w, (si, di, code),
                      (wc * cloudy).sum(axis=(1, 2))[keep])
            np.add.at(liquid_w, (si, di, code),
                      (wc * liquid).sum(axis=(1, 2))[keep])
            for q, field in values.items():
                finite = np.isfinite(field)
                np.add.at(vw[q], (si, di, code),
                          (wc * finite * np.nan_to_num(field)).sum(axis=(1, 2))[keep])
                np.add.at(vw_w[q], (si, di, code),
                          (wc * finite).sum(axis=(1, 2))[keep])

    # --- per-season slot values ---------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        area_pct = np.where(w_domain[..., None] > 0,
                            100.0 * w_class / w_domain[..., None], np.nan)
        series = {
            q: np.where(vw_w[q] > 0, vw[q] / np.where(vw_w[q] > 0, vw_w[q], 1.0),
                        np.nan)
            for q in MEAN_SOURCES
        }
        series["liquid_fraction"] = np.where(
            cloudy_w > 0, 100.0 * liquid_w / np.where(cloudy_w > 0, cloudy_w, 1.0),
            np.nan,
        )

    return {
        "series": series,
        "area_pct": area_pct,
        "slots": slots,
        "seasons": uniq_seasons,
        "n_unclassified": n_unclassified,
        "site_code": site_code,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "site_class_counts": site_class_counts,
    }


def collapse_seasons(sec: dict, keep_idx: list[int]) -> dict:
    """Average the kept seasons, equal weight each, into (slot, class) arrays."""
    out = {"slots": sec["slots"]}
    out["area_pct"] = nanmean_quiet(sec["area_pct"][keep_idx], axis=0)
    out["series"] = {
        q: nanmean_quiet(v[keep_idx], axis=0) for q, v in sec["series"].items()
    }
    # Carry the site metadata through: make_figure looks for site_code here to
    # decide whether to draw the overlay, and silently drew nothing without it.
    for k in ("site_code", "site_lat", "site_lon"):
        if k in sec:
            out[k] = sec[k]
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_figure(
    col: dict,
    quantity: str,
    region: str,
    mode_label: str,
    args,
    output_path: Path | None = None,
    dpi: int | None = None,
):
    """Draw the two-panel figure for one quantity, and save it if asked.

    ``output_path`` of None draws without writing, which is what a notebook
    wants; the figure is returned either way.
    """
    import matplotlib.pyplot as plt

    label, units, _kind = QUANTITIES[quantity]
    slots = col["slots"]
    x = np.arange(len(slots))
    field = col["series"][quantity]      # (slot, class)
    area = col["area_pct"]               # (slot, class)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )

    # --- top: the variable, one line per class ------------------------------
    for name in CLASS_ORDER:
        code = CLASS_CODES[name]
        y = field[:, code].copy()
        # Blank the line where the class is too small for its mean to mean
        # anything -- see the module docstring.
        y[area[:, code] < args.min_class_area] = np.nan
        # Smooth AFTER blanking, so a small-footprint stretch cannot leak back
        # in through the window.
        y = running_mean(y, args.smooth)
        if np.all(np.isnan(y)):
            continue
        ax_top.plot(x, y, color=CLASS_COLORS[name], linewidth=1.8,
                    label=CLASS_LABELS[name], solid_capstyle="round")

    # The ARM site cell, drawn last so it sits on top, and dashed so it reads as
    # a point comparison rather than another area class. No min_class_area test:
    # it is one cell by construction, and blanking it for being small would
    # blank it always.
    site_code = col.get("site_code")
    if site_code is not None and site_code < field.shape[1]:
        y = running_mean(field[:, site_code].copy(), args.smooth)
        if not np.all(np.isnan(y)):
            ax_top.plot(x, y, color=SITE_COLOR, linewidth=1.7, linestyle="--",
                        label=SITE_LABEL, solid_capstyle="round", zorder=5)

    ax_top.set_ylabel(f"{label} [{units}]")
    ax_top.grid(alpha=0.25, linewidth=0.6)
    ax_top.legend(loc="upper right", ncol=len(CLASS_ORDER) + 1, frameon=True,
                  framealpha=0.9, fontsize=9, columnspacing=1.1,
                  handlelength=1.6)

    # --- bottom: area share --------------------------------------------------
    # Only the five classes go in the area panel. They partition the domain and
    # sum to 100%; the site cell is already inside one of them, so including it
    # would double-count and break the stack. See SITE_KEY.
    stack = np.nan_to_num(area[:, [CLASS_CODES[n] for n in CLASS_ORDER]].T)
    if args.area_style == "stacked":
        ax_bot.stackplot(
            x, *stack,
            colors=[CLASS_COLORS[n] for n in CLASS_ORDER],
            labels=[CLASS_LABELS[n] for n in CLASS_ORDER],
            edgecolor="none",
        )
        ax_bot.set_ylim(0, 100)
    else:
        for name in CLASS_ORDER:
            ax_bot.plot(x, area[:, CLASS_CODES[name]], color=CLASS_COLORS[name],
                        linewidth=1.6, label=CLASS_LABELS[name])
        ax_bot.set_ylim(0, None)
    ax_bot.set_ylabel("Area of region [%]")
    ax_bot.set_xlabel(
        f"Day of season ({args.season_start[0]:02d}-{args.season_start[1]:02d} "
        f"to {args.season_end[0]:02d}-{args.season_end[1]:02d})"
    )
    ax_bot.grid(alpha=0.25, linewidth=0.6, color="white" if
                args.area_style == "stacked" else "grey")

    pos, lab = month_ticks(slots)
    ax_bot.set_xticks(pos)
    ax_bot.set_xticklabels(lab)
    ax_bot.set_xlim(0, len(slots) - 1)

    blanked = ""
    if col.get("site_code") is not None:
        blanked += "   |   ARM site = 1 cell, not in the area panel"
    if args.min_class_area > 0:
        # += , not = : this used to be the first clause and silently discarded
        # anything appended before it.
        blanked += f"   |   line blanked below {args.min_class_area:g}% of area"
    if args.smooth > 1:
        blanked += f"   |   {args.smooth}-day running mean"
    if quantity == "liquid_fraction":
        detail = (f"liquid = LWP > {args.lwp_threshold:g} g m$^{{-2}}$   |   "
                  f"cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}")
    else:
        detail = (f"area-weighted mean   |   trace "
                  f"$\\leq$ {args.trace_threshold:g} g m$^{{-2}}$ zeroed"
                  if quantity in ("lwp", "iwp") else "area-weighted mean")

    fig.suptitle(
        f"{label} by surface class — {region}\n"
        f"{mode_label}   |   {detail}{blanked}",
        fontsize=13, y=0.965,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.075, right=0.985)
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi or 150)
        print(f"  -> {output_path}")
    return fig


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    add_classification_args(parser)
    parser.add_argument("--variable", nargs="+",
                        choices=sorted(QUANTITIES) + ["all"], default=["all"],
                        help="Which quantity to plot, one figure each "
                             "(default: all four).")
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD", help="Season start (default 08-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(3, 31),
                        metavar="MM-DD",
                        help="Season end, inclusive (default 03-31, wrapping the "
                             "year).")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Seasons to average, by the year each season STARTS "
                             "in: '2019', '2019-2025', or '2000,2019-2020'. "
                             "Default: every season meeting --min-season-coverage.")
    parser.add_argument("--lwp-threshold", type=float,
                        default=DEFAULT_LWP_THRESHOLD_G, metavar="G",
                        help="LWP in g m-2 above which liquid counts as PRESENT, "
                             f"for the liquid_fraction panel (default "
                             f"{DEFAULT_LWP_THRESHOLD_G:g}). Matches "
                             "plot_latitude_time_hovmoller_liquid_fraction.py.")
    parser.add_argument("--iwp-threshold", type=float,
                        default=DEFAULT_IWP_THRESHOLD_G, metavar="G",
                        help="IWP in g m-2 used in the liquid_fraction panel's "
                             f"cloudy test (default {DEFAULT_IWP_THRESHOLD_G:g}). "
                             "Same unit as --lwp-threshold and --trace-threshold.")
    parser.add_argument("--min-cloud-fraction", type=float,
                        default=DEFAULT_MIN_CLOUD_FRACTION, metavar="F",
                        help="Total cloud cover at or above which a scene counts "
                             f"as cloudy (default {DEFAULT_MIN_CLOUD_FRACTION:g}, "
                             "fully overcast).")
    parser.add_argument("--trace-threshold", type=float,
                        default=DEFAULT_TRACE_THRESHOLD_G, metavar="G",
                        help="LWP/IWP in g m-2 at or below which an hour is "
                             "treated as zero before the MAGNITUDE mean (default "
                             f"{DEFAULT_TRACE_THRESHOLD_G:g}). Separate from "
                             "--lwp-threshold; matches plot_monthly_lwp_maps.py.")
    parser.add_argument("--min-class-area", type=float,
                        default=DEFAULT_MIN_CLASS_AREA_PCT, metavar="PCT",
                        help="Blank a class's line in the top panel wherever it "
                             "holds less than this %% of the domain area (default "
                             f"{DEFAULT_MIN_CLASS_AREA_PCT:g}). Set 0 to draw "
                             "everything.")
    parser.add_argument("--smooth", type=int, default=1, metavar="DAYS",
                        help="Centred running mean applied to the TOP panel only, "
                             "in days of season (default 1, i.e. none). Synoptic "
                             "variability dominates a single season; 5-11 makes "
                             "the separation between classes legible without "
                             "touching the area panel.")
    parser.add_argument("--area-style", choices=("stacked", "lines"),
                        default="stacked",
                        help="Bottom panel as a stacked area (default) or as "
                             "one line per class.")
    parser.add_argument("--min-season-coverage", type=float, default=0.6,
                        metavar="F",
                        help="Exclude seasons covering less than this fraction of "
                             "the window from a climatology (default 0.6).")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


class Analysis(SimpleNamespace):
    """Everything the figures need, computed once.

    Built by :func:`prepare`. ``build_series`` accumulates ALL four quantities
    in one streaming pass over the archive, so every figure below is already
    paid for by the time this object exists -- a notebook can load once and then
    redraw any single figure for free.
    """


def prepare(argv=None, args=None, **overrides):
    """Load the archive, classify the surface, and reduce to season arrays.

    ``argv`` takes the same strings as the command line; ``overrides`` sets
    individual options by name, e.g. ``prepare(region="barrow", years=(2019,))``.
    Returns an :class:`Analysis`.

    This is the slow step. ``--variable`` is deliberately NOT honoured here:
    every quantity is computed regardless, so the notebook's figure cells are
    all available without a reload.
    """
    if args is None:
        args = parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    print("=" * 72)
    print("Surface-class time series")
    print("=" * 72)

    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm_da = load_land_sea_mask(
        args.region, resolve_data_root(args.storage, args.data_root),
        args.mask_grid,
    )
    lsm = align_lsm_to_grid(lsm_da, ds)

    missing = sorted(set(REQUIRED_VARS) - set(ds.data_vars))
    if missing:
        raise KeyError(f"dataset is missing {missing}. Re-download with "
                       f"--var-set recommended or extended")

    print(f"  Source     : {region_dir}")
    print(f"  Grid       : {ds.sizes['latitude']} x {ds.sizes['longitude']} cells, "
          f"{ds.sizes['valid_time']:,} time steps")
    print(f"  Season     : {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
          f"{args.season_end[0]:02d}-{args.season_end[1]:02d}"
          + ("  (wraps the new year)" if args.season_end < args.season_start else ""))
    print(f"  Classes    : lsm tol {args.lsm_tol:g} | open ocean < "
          f"{args.open_ocean_max_siconc:g} | pack ice > {args.sea_ice_min_siconc:g}")

    layout = season_layout(ds, args)
    keep_idx, used, mode_label = select_seasons(layout, args)
    print(f"\n  Reading {len(used)} season(s): {used}")

    sec = build_series(ds, lsm, args, layout, keep_idx)
    if sec["n_unclassified"]:
        print(f"  !! {sec['n_unclassified']:,} unclassified cell-times; run "
              f"surface_classification.py for the breakdown.", file=sys.stderr)
    col = collapse_seasons(sec, keep_idx)

    tag = f"season{used[0]}" if len(used) == 1 else f"mean{used[0]}-{used[-1]}"
    return Analysis(args=args, ds=ds, lsm=lsm, layout=layout, keep_idx=keep_idx,
                    used=used, mode_label=mode_label, sec=sec, col=col, tag=tag)


def print_report(A):
    """Mean area share per surface class, and where the ARM site cell landed."""
    print("\n  Mean area share over the season window:")
    for name in CLASS_ORDER:
        share = nanmean_quiet(A.col["area_pct"][:, CLASS_CODES[name]])
        print(f"    {CLASS_LABELS[name]:<20}{share:6.2f}%")

    sec = A.sec
    if "site_code" not in sec:
        return
    d_lat = sec["site_lat"] - SITE_LAT
    d_lon = sec["site_lon"] - SITE_LON
    print(f"\n  {SITE_LABEL}: facility at {SITE_LAT:.3f} N, {SITE_LON:.3f} E")
    print(f"    nearest cell centre  : {sec['site_lat']:.3f} N, "
          f"{sec['site_lon']:.3f} E  (offset {d_lat:+.3f}, {d_lon:+.3f} deg)")
    share = nanmean_quiet(A.col["area_pct"][:, sec["site_code"]])
    print(f"    share of the domain  : {share:.3f}%  (one cell; not stacked)")
    counts = sec["site_class_counts"]
    total = int(counts.sum())
    if total:
        print("    the cell classifies as, over the window:")
        for name in CLASS_ORDER:
            n = int(counts[CLASS_CODES[name]])
            if n:
                print(f"      {CLASS_LABELS[name]:<20}{100*n/total:6.2f}%")
        if counts[-1]:
            print(f"      {'Unclassified':<20}{100*counts[-1]/total:6.2f}%")


def figure(A, quantity, out_dir=None, dpi=None):
    """Draw one quantity's two-panel figure from a prepared :class:`Analysis`.

    Saves into ``out_dir`` when one is given, and always returns the figure.
    """
    if quantity not in QUANTITIES:
        raise KeyError(f"unknown quantity {quantity!r}; "
                       f"choose from {sorted(QUANTITIES)}")
    path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{A.args.region}_surfaceclass_{quantity}_{A.tag}.png"
    return make_figure(A.col, quantity, A.args.region, A.mode_label, A.args,
                       path, dpi or A.args.dpi)


# ----------------------------------------------------------------------------
# Monthly summary across classes
# ----------------------------------------------------------------------------
def monthly_medians(A):
    """Median monthly value of every quantity, per class and for the ARM cell.

    Returns ``(months, {quantity: array[month, series]})`` where the series axis
    is ``CLASS_ORDER`` followed by the ARM site.

    TWO NESTED MEDIANS, in this order:

      1. within a season, the median over the day-slots of that month
      2. across seasons, the median of those monthly values

    Season-first is deliberate. Collapsing seasons before taking the monthly
    median would let one anomalous year drag the month, which in this region it
    does -- December 2022 held 50% marginal ice where 2025 held 3%. Two nested
    medians keep a single odd season from setting the answer.

    The same ``--min-class-area`` blanking the line figures use is applied
    first, so a class too small for its daily mean to mean anything does not
    contribute to its monthly median either. The ARM site is exempt: it is one
    cell by construction and would be blanked everywhere.
    """
    slots = A.col["slots"]
    mon_of_slot = np.array([m for m, _ in slots])
    months = list(dict.fromkeys(mon_of_slot.tolist()))
    codes = [CLASS_CODES[n] for n in CLASS_ORDER]
    site_code = A.sec.get("site_code")
    if site_code is not None:
        codes = codes + [site_code]

    area = A.sec["area_pct"]                      # (season, slot, class)
    out = {}
    for q, arr in A.sec["series"].items():        # (season, slot, class)
        vals = np.full((len(months), len(codes)), np.nan)
        for ci, code in enumerate(codes):
            block = arr[A.keep_idx][:, :, code].copy()        # (season, slot)
            if code != site_code and A.args.min_class_area > 0:
                small = area[A.keep_idx][:, :, code] < A.args.min_class_area
                block[small] = np.nan
            for mi, m in enumerate(months):
                sel = mon_of_slot == m
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    per_season = np.nanmedian(block[:, sel], axis=1)
                    vals[mi, ci] = np.nanmedian(per_season)
        out[q] = vals
    return months, out


def _series_style(A):
    """(label, colour) for each series drawn by the monthly figures."""
    style = [(CLASS_LABELS[n], CLASS_COLORS[n]) for n in CLASS_ORDER]
    if A.sec.get("site_code") is not None:
        style.append((SITE_LABEL, SITE_COLOR))
    return style


def _monthly_panels(A, title, subtitle):
    """Shared 2x2 canvas for the monthly comparison figures."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    fig.suptitle(f"{title} \u2014 {A.args.region}\n{A.mode_label}   |   {subtitle}",
                 fontsize=13)
    return fig, axes.ravel()


def _finish_panel(ax, q, months, legend=False, style=None):
    label, units, _ = QUANTITIES[q]
    ax.set_ylabel(f"{label} [{units}]", fontsize=10)
    ax.set_xticks(np.arange(len(months)))
    ax.set_xticklabels([calendar.month_abbr[m] for m in months])
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if legend and style:
        ax.legend(fontsize=8, framealpha=0.9, ncol=2, handlelength=1.4,
                  columnspacing=1.0)


def fig_monthly_grouped(A, out_dir=None, dpi=None):
    """Grouped bars: one bar per series, per month, per quantity."""
    months, med = monthly_medians(A)
    style = _series_style(A)
    n = len(style)
    fig, axes = _monthly_panels(
        A, "Monthly median by surface class",
        "median over days in month, then over seasons")
    x = np.arange(len(months))
    width = 0.84 / n
    for ax, q in zip(axes, sorted(QUANTITIES)):
        for i, (lab, color) in enumerate(style):
            off = (i - (n - 1) / 2) * width
            ax.bar(x + off, med[q][:, i], width=width * 0.9, color=color,
                   label=lab, edgecolor="white", linewidth=0.3)
        _finish_panel(ax, q, months, legend=(q == sorted(QUANTITIES)[0]),
                      style=style)
    return _emit_monthly(fig, A, out_dir, "monthly_grouped", dpi)


def fig_monthly_stacked(A, out_dir=None, dpi=None):
    """Stacked bars. SHOWN FOR COMPARISON; the stack total is not a quantity.

    Requested as an alternative to the grouped view, and drawn faithfully, but
    it should not be used to read anything off. Stacking asserts that the
    segments add up to a meaningful whole; these six numbers are independent
    estimates of the SAME variable over different subsets of the domain, so
    their sum has no physical meaning and the bar height is an artefact of how
    many classes happen to be plotted. A reader also cannot compare segments
    across months, because only the bottom segment shares a baseline. Use
    :func:`fig_monthly_grouped` or :func:`fig_monthly_dots` instead.
    """
    months, med = monthly_medians(A)
    style = _series_style(A)
    fig, axes = _monthly_panels(
        A, "Monthly median by surface class, STACKED",
        "segments do NOT sum to a physical total \u2014 see the docstring")
    x = np.arange(len(months))
    for ax, q in zip(axes, sorted(QUANTITIES)):
        bottom = np.zeros(len(months))
        for i, (lab, color) in enumerate(style):
            h = np.nan_to_num(med[q][:, i])
            ax.bar(x, h, width=0.62, bottom=bottom, color=color, label=lab,
                   edgecolor="white", linewidth=0.4)
            bottom += h
        _finish_panel(ax, q, months, legend=(q == sorted(QUANTITIES)[0]),
                      style=style)
        ax.set_ylabel(ax.get_ylabel() + "\n(stacked sum)", fontsize=9)
    return _emit_monthly(fig, A, out_dir, "monthly_stacked", dpi)


def fig_monthly_dots(A, out_dir=None, dpi=None):
    """All six values at one x position per month, on a shared baseline.

    What the stacked view was reaching for -- six comparable numbers in one
    slot -- without the false claim that they add up. Each month is a single
    tick; the six markers sit at their actual values, so both within-month and
    across-month comparison read directly off the y axis.
    """
    months, med = monthly_medians(A)
    style = _series_style(A)
    n = len(style)
    fig, axes = _monthly_panels(
        A, "Monthly median by surface class",
        "one slot per month, six values on a shared axis")
    x = np.arange(len(months))
    width = 0.66
    for ax, q in zip(axes, sorted(QUANTITIES)):
        lo = np.nanmin(med[q]) if np.isfinite(med[q]).any() else 0.0
        for xi in x:
            ax.plot([xi - width / 2, xi + width / 2], [lo, lo], color="0.85",
                    lw=0.8, zorder=0)
        for i, (lab, color) in enumerate(style):
            off = (i - (n - 1) / 2) * (width / n)
            y = med[q][:, i]
            ax.vlines(x + off, lo, y, color=color, lw=1.4, alpha=0.55, zorder=2)
            ax.plot(x + off, y, "o", color=color, ms=6, label=lab,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        _finish_panel(ax, q, months, legend=(q == sorted(QUANTITIES)[0]),
                      style=style)
    return _emit_monthly(fig, A, out_dir, "monthly_dots", dpi)


def _emit_monthly(fig, A, out_dir, stem, dpi):
    if out_dir is None:
        return fig
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{A.args.region}_surfaceclass_{stem}_{A.tag}.png"
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def fig_dlr(A, out_dir=None, dpi=None):
    """Downwelling longwave by surface class."""
    return figure(A, "dlr", out_dir, dpi)


def fig_lwp(A, out_dir=None, dpi=None):
    """Liquid water path by surface class."""
    return figure(A, "lwp", out_dir, dpi)


def fig_iwp(A, out_dir=None, dpi=None):
    """Ice water path by surface class."""
    return figure(A, "iwp", out_dir, dpi)


def fig_liquid_fraction(A, out_dir=None, dpi=None):
    """Share of cloudy scenes containing liquid, by surface class."""
    return figure(A, "liquid_fraction", out_dir, dpi)


ALL_FIGURES = (fig_dlr, fig_lwp, fig_iwp, fig_liquid_fraction)

# The monthly summaries are not in ALL_FIGURES: they are a different
# view of the same numbers, and fig_monthly_stacked is a deliberate
# counter-example rather than something to write out by default.
MONTHLY_FIGURES = (fig_monthly_grouped, fig_monthly_dots)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quantities = (sorted(QUANTITIES) if "all" in args.variable
                  else list(dict.fromkeys(args.variable)))
    try:
        A = prepare(args=args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    print(f"  Quantities : {', '.join(quantities)}")
    print_report(A)

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    print()
    for quantity in quantities:
        figure(A, quantity, out_dir=out_dir)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
