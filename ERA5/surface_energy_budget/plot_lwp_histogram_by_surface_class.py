#!/usr/bin/env python3
"""Hours per season with a liquid-bearing cloud, binned by LWP, per surface class.

One panel per surface class from ``surface_classification.py`` -- land, coastal,
open ocean, marginal ice, sea ice -- plus a sixth panel for the single ERA5 grid
cell holding the DOE ARM facility at Utqiagvik. Inside each panel the bars are
stacked by cloud phase from ``cloud_classification.py``: liquid-only clouds at
the bottom, mixed-phase on top, so the full bar height is "a cloud containing
liquid" and the split says how much of it also carried ice.

The figure answers, for each surface type: over a season, how many hours does a
grid cell of that type spend under a liquid-bearing cloud, and how is that time
distributed over liquid water path?

TWO COPIES OF EVERY FIGURE
==========================
Each run writes the same histogram twice, differing in how the LWP axis is
binned:

    linear   equal-width bins, 0 to --lwp-lin-max g m-2. Reads like a physical
             axis and is directly comparable to a linear-axis figure from a
             ground site, but puts most of the winter population in bar 1.
    log      log-spaced bins, by default from --lwp-min (the liquid-bearing
             floor, below which there is nothing to draw) to --lwp-log-max.
             Resolves the low-LWP end, where the midwinter population lives.

Both are drawn from the SAME cell-hours -- only the bin edges differ -- so bar
totals agree between them to the last hour. Neither is a smoothed or fitted
version of the other.

``--y-scale log`` additionally puts the HOURS axis on a log scale, which is a
different question (the tail of rare high-LWP hours) and is off by default.

WHAT ONE BAR MEANS, AND WHAT THE HOURS ARE PER
==============================================
The unit is the **cell-hour**: one 0.25 deg grid cell at one hourly time step.
A bar height is

    hours per season = ( area-weighted cell-hours of this class in this LWP bin
                         and this phase )
                     / ( area-weighted cell-hours of this class, any sky )
                     x ( hours in the full season window )

which is a per-CELL rate, not a total over the class. That normalisation is what
makes the six panels comparable at all: without it the sea-ice panel would tower
over the Utqiagvik panel purely because it holds a few thousand times more
cells, which says nothing about clouds.

Read a bar as: "a cell of this surface type spends N hours of the season under a
liquid cloud whose LWP falls in this bin." Summing every bar in a panel gives
the class's total liquid-cloud hours per season, which is printed in the panel
legend. The season window itself is len(--season-start .. --season-end) x 24 h --
5,856 h for the default 1 August to 31 March.

CLASS MEMBERSHIP MOVES IN TIME, AND THE DENOMINATOR MOVES WITH IT
-----------------------------------------------------------------
Three of the five classes are defined by sea ice concentration, so a cell is
open ocean in September and sea ice in February. Both numerator and denominator
above are accumulated over exactly the cell-hours the cell spent IN the class,
so the ratio is the class's own conditional occupancy and the seasonal migration
of the ice edge does not leak into it. The consequence worth stating plainly:
the "open ocean" panel is not a season-long time series of fixed cells, it is
the pooled behaviour of whatever water was ice-free at the time.

PARTIAL SEASONS ARE SCALED UP, NOT COUNTED SHORT
------------------------------------------------
The denominator runs over the hours actually present in the archive, and the
result is then multiplied by the NOMINAL window length. A season missing 10% of
its hours is therefore reported as though those hours behaved like the ones that
were sampled, rather than as a season that was 10% less cloudy. That is the
right default for a climatology and the wrong one if the gap is systematic -- a
season missing all of February is not well described by its August-January rate.
``--min-season-coverage`` guards this, and every season's coverage is printed
before anything is read.

CLOUD PHASE
===========
Masks come from ``cloud_classification.cloud_phase_masks``, applied to scenes
that already pass the cloud-cover test:

    cloudy         tcc >= --min-cloud-fraction       (default 1.0, overcast)
    liquid-bearing cloudy and LWP > --lwp-min        (default 5 g m-2)
      liquid only    ... and IWP < --iwp-min         the lower stack segment
      mixed phase    ... and IWP > --iwp-min         the upper stack segment
    ice only       cloudy and IWP > --iwp-min and LWP < --lwp-max-ice

The full height of a bar is the liquid-bearing hours; the colour split is the
single ice threshold --iwp-min. The two segments therefore PARTITION the bar and
sum to the total printed in the panel title, which is the only way a stacked bar
can be read honestly. (``cloud_classification`` leaves a deliberate gap between
its liquid ceiling and its mixed floor; ``--iwp-max-liquid`` reopens that gap
here if you want it, and the hours that fall into it are then reported rather
than silently lost between the segments.)

Ice-only scenes are NOT drawn. An ice cloud has no liquid water path to bin, so
every one of them would pile into the first bar and say nothing about LWP. They
are in the text summary instead, because how much of a class's cloudy time is
ice-only is exactly the context the figure lacks.

WHY THE THRESHOLD DEFAULTS ARE NOT cloud_classification's
---------------------------------------------------------
``cloud_classification`` defaults --lwp-min and --iwp-min to 0.03 g m-2, ERA5's
trace quantum. At that level essentially every overcast Arctic column holds a
trace of both species, so "mixed phase" degenerates into "overcast" and the
stack collapses to one colour. MEASURED, Barrow strip, season 2020/21, open
ocean: of 3,546 overcast hours per cell, 3,446 classified as mixed and 96 as
liquid only. This script therefore defaults both floors to 5 g m-2, the value
``plot_surface_class_timeseries.py`` already uses for the same purpose. Raise
--lwp-min to 10-25 to sit near a microwave radiometer's detection floor if the
figure is going to be compared against a ground-based retrieval.

Note the two cloud-cover conventions in this directory. ``cloud_classification``
defines ``cloudy`` as ``tcc > --cloudy-threshold`` with a 0.5 default; the SEB
plotting scripts use ``tcc >= --min-cloud-fraction`` with a 1.0 default. This
script follows the plotting scripts, so its numbers line up with
``plot_surface_class_timeseries.py`` and ``analyze_cloud_liquid_frequency.py``
without a threshold conversion in your head. Pass ``--min-cloud-fraction 0`` to
drop the cloud-cover gate entirely and classify on condensate alone.

UNITS
=====
Every path on the command line and in this file is **g m-2**. ERA5 stores
``tclw``/``tciw`` in kg m-2; the conversion happens once, at the point each block
is read.

THE MEDIAN LINE
===============
The dotted line in each panel is the area-weighted median LWP over that class's
liquid-bearing cloud hours (both phases pooled). It is computed from a fine
log-spaced accumulator, independent of the display bins, so it does not move
between the linear and log copies of the figure.

OPTIONS
=======
Data source
-----------
--storage {local,external}   Which disk to read (default local).
--data-root PATH             Explicit directory, overriding --storage.
--region NAME                Region subdirectory (default barrow).

Season and years
----------------
--season-start MM-DD         First day of the window (default 08-01).
--season-end MM-DD           Last day, inclusive (default 03-31, wraps the year).
--years SPEC                 '2019', '2019-2025', or '2000,2019-2020', by the
                             year each season STARTS in. Default: every season
                             meeting --min-season-coverage.
--min-season-coverage F      Drop seasons covering less of the window than this
                             (default 0.6). An explicit --years is honoured
                             anyway, with a warning.

Cloud phase (all in g m-2)
-------------------------
--lwp-min G                  Liquid-bearing floor, the full bar (default 5).
--iwp-min G                  Mixed-phase floor, the colour split (default 5).
--iwp-max-liquid G           Liquid-only ceiling (default: equal to --iwp-min).
--lwp-max-ice G              Ice-only ceiling (default 0.001). Report only.
--min-cloud-fraction F       Cloud cover for a cloudy scene (default 1.0).

Binning
-------
--lwp-lin-max G              Top of the linear axis (default 600).
--lwp-lin-bins N             Linear bins below it (default 24, i.e. 25 g m-2).
--lwp-log-min G              First log edge (default: equal to --lwp-min).
--lwp-log-max G              Last log edge (default 1000).
--lwp-log-bins N             Log bins between them (default 12).
--bin-scale {linear,log,both}  Which copies to write (default both).
--y-scale {linear,log}       Scale of the hours axis (default linear).

Examples
--------
    ./plot_lwp_histogram_by_surface_class.py --region barrow --years 2015-2025

    ./plot_lwp_histogram_by_surface_class.py --region barrow --years 2015-2025 \
        --season-start 11-01 --season-end 02-28 --lwp-min 5

Requires the mask from ``download_era5_land_sea_mask.py`` for the region.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cloud_classification import (
    DEFAULT_LWP_MAX_ICE_G,
    PHASE_COLORS,
    PHASE_LABELS,
    cloud_phase_masks,
)
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
)
from surface_classification import (
    CLASS_CODES,
    CLASS_LABELS,
    CLASS_ORDER,
    DEFAULT_BLOCK_HOURS,
    UNCLASSIFIED,
    add_classification_args,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

# The ARM site cell and the season bookkeeping are shared with the time-series
# script rather than reimplemented, so the two figures can never disagree about
# which cell Utqiagvik is or which seasons a given --years selects.
from plot_surface_class_timeseries import (
    SITE_COLOR,
    SITE_LABEL,
    SITE_LAT,
    SITE_LON,
    nanmean_quiet,
    parse_month_day,
    parse_years,
    season_layout,
    select_seasons,
    site_cell_mask,
)

# ERA5 hourly data: one time step is one hour, so a cell-hour count IS an hour
# count and no conversion appears anywhere below.
HOURS_PER_STEP = 1.0

REQUIRED_VARS = ("tcc", "tclw", "tciw", "siconc")

# Phases drawn, in stack order from the bottom. "ice" is accumulated as well but
# never plotted -- it has no liquid water path to bin. See the module docstring.
PHASE_STACK: tuple[str, ...] = ("liquid", "mixed")
PHASE_ORDER_ACC: tuple[str, ...] = ("liquid", "mixed", "ice")

# Cloud-cover gate, matching plot_surface_class_timeseries.py and
# analyze_cloud_liquid_frequency.py rather than cloud_classification's
# --cloudy-threshold. See the module docstring.
DEFAULT_MIN_CLOUD_FRACTION = 1.0

# Phase thresholds, g m-2. These are the SEMANTICS of cloud_classification.py --
# the masks come from its cloud_phase_masks -- but not its defaults, and the
# departure is deliberate on both counts.
#
# 1. 5 g m-2 rather than the 0.03 g m-2 trace quantum. MEASURED on the Barrow
#    strip, season 2020/21, at the trace defaults: of 3,546 overcast hours per
#    cell over open ocean, 3,446 came out MIXED and 96 liquid-only. ERA5 carries
#    a trace of both species in nearly every overcast column, so a trace-level
#    ice floor makes "mixed phase" mean "overcast", the stack collapses to one
#    colour, and the figure loses the distinction it exists to draw. 5 g m-2 is
#    the value plot_surface_class_timeseries.py already uses for the same job.
# 2. The liquid-only CEILING is tied to the mixed-phase FLOOR (see
#    resolve_phase_thresholds). cloud_classification leaves a deliberate gap
#    between them, which is right when classifying every scene into three
#    labelled bins and wrong here: a stacked bar whose segments do not partition
#    the population it is labelled with silently drops hours between the
#    segments and the total printed above them.
DEFAULT_LWP_MIN_G = 5.0
DEFAULT_IWP_MIN_G = 5.0

# ERA5 packs tclw/tciw with a GRIB binary scale factor of 2**-15 kg m-2, so the
# smallest non-zero path the archive can express is this. An ice threshold set
# within a quantum or two of it does not separate "no ice" from "some ice" -- it
# separates "literally zero ice" from everything else, and in an overcast Arctic
# column that essentially never happens. MEASURED, Barrow strip, November 2022,
# tcc >= 0.99, LWP > 0.031 g m-2: 99.92% of those cell-hours carry at least two
# quanta of ice, so a 0.031 g m-2 ice threshold leaves "liquid only" holding
# 0.08% of the liquid-bearing hours -- one or two hours a season.
ERA5_PATH_QUANTUM_G = 2.0 ** -15 * 1000.0        # 0.0305176 g m-2

# A stacked segment below this share of the liquid-bearing hours has collapsed:
# the figure is one colour and the split conveys nothing. Not an error -- the
# thresholds were honoured exactly -- so it is reported, not raised.
DEGENERATE_SEGMENT_SHARE = 0.01

# Display bin defaults. The linear set reproduces a 0-600 g m-2 axis in
# 25 g m-2 steps; the log set matches analyze_cloud_liquid_frequency.py.
DEFAULT_LWP_LIN_MAX_G = 600.0
DEFAULT_LWP_LIN_BINS = 24
# None means "start the log axis at --lwp-min". Below that floor there are no
# liquid-bearing hours by definition, so a fixed 0.1 g m-2 default would spend
# most of the axis on bins that cannot contain anything.
DEFAULT_LWP_LOG_MIN_G = None
DEFAULT_LWP_LOG_MAX_G = 1000.0
DEFAULT_LWP_LOG_BINS = 12

# A class holding less than this share of the domain area gets a warning stamped
# on its panel: with a handful of cells, a "typical cell of this class" is not a
# meaningful object.
DEFAULT_MIN_CLASS_AREA_PCT = 0.5

# Fine, display-independent grid used only for the median line and the reported
# quantiles: 9 decades at 200 bins per decade, i.e. ~1.2% bin width, so a median
# read off it is exact to well under the width of any display bar.
QUANTILE_EDGES_G = np.geomspace(1e-4, 1e5, 1801)

DEFAULT_LAYOUT = (2, 3)


# ----------------------------------------------------------------------------
# Cloud phase
# ----------------------------------------------------------------------------
def resolve_phase_thresholds(args) -> dict:
    """The four thresholds actually handed to ``cloud_phase_masks``, in g m-2.

    ``--iwp-max-liquid`` defaults to ``--iwp-min`` rather than to
    cloud_classification's near-zero ceiling, so that

        liquid only   LWP > --lwp-min  and  IWP < --iwp-min
        mixed phase   LWP > --lwp-min  and  IWP > --iwp-min

    PARTITION the liquid-bearing hours between them. That is what lets a stacked
    bar be read as a whole: its two segments sum to the total in the panel title
    with nothing lost in between. Passing ``--iwp-max-liquid`` explicitly
    reopens the gap, and the hours that fall into it are then reported.

    An hour with IWP EXACTLY equal to the threshold belongs to neither, since
    both tests are strict. ERA5 quantises the paths to multiples of
    2**-15 kg m-2 = 0.0305176 g m-2, and 5 g m-2 is not one of them, so at the
    default this cannot happen; at a threshold that IS on the quantisation grid
    it costs at most one quantum's worth of hours.

    ``--lwp-max-ice`` is left at cloud_classification's near-zero default, so
    the ice-only population the report prints means "not a single quantum of
    liquid" rather than "less liquid than the mixed-phase floor".
    """
    return {
        "lwp_min_g": float(args.lwp_min),
        "iwp_min_g": float(args.iwp_min),
        "lwp_max_ice_g": float(args.lwp_max_ice),
        "iwp_max_liquid_g": float(args.iwp_min if args.iwp_max_liquid is None
                                  else args.iwp_max_liquid),
    }


def phase_split_warning(col: dict) -> str | None:
    """Message naming a collapsed stack segment, or None when the split is real.

    Guards the one failure mode of this figure that produces a plausible-looking
    picture rather than an error: an ice threshold so close to ERA5's own
    quantisation floor that "liquid only" means "not a single quantum of ice",
    which nearly nothing satisfies. The bars are then a single colour and the
    legend reports a handful of hours, with no indication that the threshold and
    not the atmosphere is responsible.

    Summed over the five classes only. The ARM site cell is inside one of them,
    so including it would count its hours twice.
    """
    mean_h = col["hours"]["linear"]["mean"]              # (class, phase, bar)
    pk = col["phase_kw"]
    codes = [CLASS_CODES[n] for n in CLASS_ORDER]
    i_liq = PHASE_ORDER_ACC.index("liquid")
    i_mix = PHASE_ORDER_ACC.index("mixed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        liq = float(np.nansum(mean_h[codes, i_liq]))
        mix = float(np.nansum(mean_h[codes, i_mix]))
    total = liq + mix
    if total <= 0:
        return None

    for name, share, other in (("liquid only", liq / total, "mixed phase"),
                               ("mixed phase", mix / total, "liquid only")):
        if share >= DEGENERATE_SEGMENT_SHARE:
            continue
        msg = (f"the '{name}' segment holds {100 * share:.3f}% of the "
               f"liquid-bearing hours, so the stack is effectively all {other}")
        if name == "liquid only" and pk["iwp_min_g"] <= 3 * ERA5_PATH_QUANTUM_G:
            msg += (f". --iwp-min {pk['iwp_min_g']:g} g m-2 is within three "
                    f"quanta of ERA5's {ERA5_PATH_QUANTUM_G:.4f} g m-2 floor, "
                    f"so 'liquid only' is asking for literally zero ice. Try "
                    f"--iwp-min {DEFAULT_IWP_MIN_G:g}")
        elif name == "mixed phase":
            msg += (f". --iwp-min {pk['iwp_min_g']:g} g m-2 may be high enough "
                    f"that almost no cloud reaches it")
        return msg
    return None


# ----------------------------------------------------------------------------
# Bin construction and bar geometry
# ----------------------------------------------------------------------------
def linear_bin_edges(hi_g: float, n_bins: int) -> np.ndarray:
    """Equal-width interior edges from 0 to ``hi_g``, in g m-2."""
    if hi_g <= 0 or n_bins < 1:
        raise ValueError("--lwp-lin-max must be > 0 and --lwp-lin-bins >= 1")
    return np.linspace(0.0, float(hi_g), int(n_bins) + 1)


def log_bin_edges(lo_g: float, hi_g: float, n_bins: int) -> np.ndarray:
    """Log-spaced interior edges, in g m-2.

    Log rather than linear because the distribution spans four decades and piles
    up at the bottom: in cloudy scenes the median LWP over this domain runs from
    order 100 g m-2 in August to order 0.1 g m-2 in January. Linear bins wide
    enough for August put every winter hour in the first bar.
    """
    if lo_g <= 0 or hi_g <= lo_g or n_bins < 1:
        raise ValueError("need 0 < --lwp-log-min < --lwp-log-max and "
                         "--lwp-log-bins >= 1")
    return np.geomspace(float(lo_g), float(hi_g), int(n_bins) + 1)


def n_bars(edges_g: np.ndarray) -> int:
    """Bar count for an edge set: underflow + interior + overflow.

    Bar 0 holds everything below ``edges_g[0]`` and bar ``len(edges_g)``
    everything at or above ``edges_g[-1]``, which is exactly what
    ``np.digitize`` returns, so no index arithmetic is needed at accumulation
    time. The linear edge set starts at 0 and its bar 0 is therefore always
    empty; the plotting code drops it rather than drawing a zero bar.
    """
    return len(edges_g) + 1


def value_to_bar_x(value_g: float, edges_g: np.ndarray, log_spaced: bool):
    """Where an LWP value sits on the categorical bar axis, or None if not finite.

    Bar ``i`` occupies [i - 0.5, i + 0.5], so interior edge ``j`` sits at
    ``j + 0.5``. Inside a bin the position is interpolated -- on the log axis
    when the bins are log-spaced -- so the median line lands where the eye
    expects it instead of snapping to a bar centre.
    """
    if value_g is None or not np.isfinite(value_g):
        return None
    if value_g < edges_g[0]:
        return 0.0
    if value_g >= edges_g[-1]:
        return float(len(edges_g))
    i = int(np.digitize(value_g, edges_g))          # edges[i-1] <= v < edges[i]
    lo, hi = float(edges_g[i - 1]), float(edges_g[i])
    if log_spaced:
        frac = (np.log10(value_g) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    else:
        frac = (value_g - lo) / (hi - lo)
    return (i - 0.5) + float(frac)


# Minimum gap, in bar widths, between two tick labels. Below this they collide.
MIN_TICK_SEPARATION = 0.6


def nice_log_values(lo_g: float, hi_g: float) -> list[float]:
    """1-2-5 decade values strictly inside ``(lo_g, hi_g)``.

    Placed by VALUE rather than by bin edge, because the log edges are only
    round numbers when the range happens to start on a decade. With the default
    axis running from --lwp-min (5 g m-2) the edges are 5, 7.775, 12.09, ...,
    and an edge-based rule labels nothing at all.
    """
    out: list[float] = []
    k = int(np.floor(np.log10(lo_g)))
    while 10.0 ** k <= hi_g:
        for mantissa in (1.0, 2.0, 5.0):
            v = mantissa * 10.0 ** k
            if lo_g < v < hi_g:
                out.append(v)
        k += 1
    return out


def nice_linear_values(lo_g: float, hi_g: float, target: int = 6) -> list[float]:
    """Round values strictly inside ``(lo_g, hi_g)``, about ``target`` of them.

    The step is taken from the 1, 2, 2.5, 5 sequence so the labels stay round
    even when --lwp-lin-max is not, which the bin edges themselves would not be.
    """
    span = hi_g - lo_g
    if span <= 0:
        return []
    raw = span / max(1, target)
    decade = 10.0 ** np.floor(np.log10(raw))
    step = next((m * decade for m in (1.0, 2.0, 2.5, 5.0) if raw <= m * decade),
                10.0 * decade)
    first = np.ceil(lo_g / step) * step
    out = []
    v = first
    while v < hi_g:
        if v > lo_g:
            out.append(float(v))
        v += step
    return out


def bar_ticks(edges_g: np.ndarray, log_spaced: bool, has_underflow: bool):
    """Tick positions and labels for the categorical bar axis.

    Labels sit at round LWP VALUES, interpolated onto the bar coordinate by
    ``value_to_bar_x``, rather than at whichever bin edges happen to be round.
    That keeps the axis readable for any --lwp-min / --lwp-lin-max the caller
    picks, instead of only for ranges that start on a decade.

    The TOP edge deliberately gets no tick of its own. The overflow bar sits
    immediately beside it and its ">" label already names that edge, so ticking
    both puts "600" and ">600" half a bar apart. The FIRST edge is named either
    by the underflow bar's "<" label or, when there is no underflow bar, by a
    tick at the left face of bar 1 -- which is where the axis begins.
    """
    lo, hi = float(edges_g[0]), float(edges_g[-1])
    if has_underflow:
        ticks, labels = [0.0], [f"<{lo:g}"]
    else:
        ticks, labels = [0.5], [f"{lo:g}"]

    values = (nice_log_values(lo, hi) if log_spaced
              else nice_linear_values(lo, hi))
    for v in values:
        x = value_to_bar_x(v, edges_g, log_spaced)
        if x is None or x - ticks[-1] < MIN_TICK_SEPARATION:
            continue
        ticks.append(x)
        labels.append(f"{v:g}")

    over_x = float(len(edges_g))
    if over_x - ticks[-1] < MIN_TICK_SEPARATION:
        ticks.pop()
        labels.pop()
    ticks.append(over_x)
    labels.append(f">{hi:g}")
    return ticks, labels


def weighted_median_from_bins(counts: np.ndarray, edges_g: np.ndarray) -> float:
    """Weighted median of a histogram whose bar 0/-1 are under/overflow.

    Interpolates linearly in log10(LWP) inside the containing bin, matching the
    log spacing of ``QUANTILE_EDGES_G``. A median that lands in either tail bar
    is reported as that bar's finite edge, since the bin has no other side; with
    the 1e-4 to 1e5 g m-2 span used here that cannot happen for real ERA5 data.
    """
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, 0.5 * total, side="left"))
    if i == 0:
        return float(edges_g[0])
    if i >= len(edges_g):
        return float(edges_g[-1])
    lo, hi = float(edges_g[i - 1]), float(edges_g[i])
    before = float(cum[i - 1])
    in_bin = float(counts[i])
    if in_bin <= 0:
        return lo
    frac = (0.5 * total - before) / in_bin
    return float(10.0 ** (np.log10(lo) + frac * (np.log10(hi) - np.log10(lo))))


# ----------------------------------------------------------------------------
# Reduction
# ----------------------------------------------------------------------------
def build_histograms(ds, lsm: np.ndarray, args, layout: dict,
                     wanted_idx: list[int], edge_sets: dict,
                     phase_kw: dict) -> dict:
    """Accumulate the per-season LWP histograms in one streaming pass.

    Both bin scales and every phase are filled from the same blocks, so the
    archive is read once no matter how many copies of the figure are wanted.
    Only the seasons in ``wanted_idx`` are read at all.

    Returns a dict of arrays indexed ``[season, class, phase, bar]`` plus the
    denominators and diagnostics the report and the figures need.
    """
    slots = layout["slots"]
    # No day-of-season axis here: the histogram pools the whole window, so
    # layout["dos"] is only needed via in_window, which already encodes it.
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    uniq_seasons = layout["seasons"]

    wanted = np.zeros(len(uniq_seasons), dtype=bool)
    wanted[wanted_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    # The ARM site rides along as one extra slot on the class axis. It is NOT a
    # sixth class -- it is already inside whichever class it falls in -- so it
    # never enters an area share that is meant to sum to 100%.
    site_mask, site_lat, site_lon = site_cell_mask(ds)
    site_code = len(CLASS_ORDER)
    n_class = len(CLASS_ORDER) + 1
    n_phase = len(PHASE_ORDER_ACC)
    n_season = len(uniq_seasons)

    hist = {
        scale: np.zeros((n_season, n_class, n_phase, n_bars(edges)))
        for scale, edges in edge_sets.items()
    }
    qhist = np.zeros((n_season, n_class, len(QUANTILE_EDGES_G) + 1))

    # Denominators and context, all area-weighted cell-hours.
    w_class = np.zeros((n_season, n_class))       # class present at all
    w_domain = np.zeros(n_season)                 # every cell, every step
    w_cloudy = np.zeros((n_season, n_class))      # class and cloudy
    w_steps = np.zeros(n_season)                  # time steps read per season

    weights_2d = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])
    w_per_step = float(weights_2d.sum())

    n_unclassified = 0
    site_class_counts = np.zeros(len(CLASS_ORDER) + 1, dtype=np.int64)

    for i0, block in iter_time_blocks(ds, list(REQUIRED_VARS), args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        sl = slice(i0, i0 + n_t)
        keep = use_step[sl]
        if not keep.any():
            continue
        si = s_idx[sl][keep]

        siconc = block["siconc"].values
        classes = classify_cells(
            lsm, siconc, args.lsm_tol, args.open_ocean_max_siconc,
            args.sea_ice_min_siconc, args.land_max_siconc,
        )[keep]
        n_unclassified += int((classes == UNCLASSIFIED).sum())

        site_codes = classes[:, site_mask]                  # (n_kept, 1)
        for code in range(len(CLASS_ORDER)):
            site_class_counts[code] += int((site_codes == code).sum())
        site_class_counts[-1] += int((site_codes == UNCLASSIFIED).sum())

        tclw_g = block["tclw"].values[keep] * 1000.0        # kg m-2 -> g m-2
        tciw_g = block["tciw"].values[keep] * 1000.0
        tcc = block["tcc"].values[keep]

        valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw_g)
        cloudy = valid & (tcc >= args.min_cloud_fraction)
        phases = cloud_phase_masks(tclw_g, tciw_g, **phase_kw)

        w = np.broadcast_to(weights_2d, classes.shape)
        np.add.at(w_domain, si, w_per_step)
        np.add.at(w_steps, si, 1.0)

        # Digitize once per scale over the whole block; the phase and class
        # masks then only select entries out of it. np.digitize already puts
        # under- and overflow in bars 0 and len(edges), which is why n_bars is
        # defined the way it is.
        bar_of = {scale: np.digitize(tclw_g, edges)
                  for scale, edges in edge_sets.items()}
        qbar = np.digitize(tclw_g, QUANTILE_EDGES_G)
        n_qbar = len(QUANTILE_EDGES_G) + 1
        # Season index broadcast to the cell grid, so a (season, bar) pair can
        # be folded into ONE flat index and accumulated with np.bincount.
        # np.add.at is the obvious call here and is roughly fifty times slower,
        # because it is unbuffered; over a multi-decade record that is the
        # difference between minutes and an afternoon.
        si_grid = np.broadcast_to(si[:, None, None], classes.shape)

        selectors = [(CLASS_CODES[name], classes == CLASS_CODES[name])
                     for name in CLASS_ORDER]
        selectors.append((site_code, np.broadcast_to(site_mask, classes.shape)))

        for code, in_class in selectors:
            wc = np.where(in_class & valid, w, 0.0)
            np.add.at(w_class, (si, code), wc.sum(axis=(1, 2)))
            np.add.at(w_cloudy, (si, code), (wc * cloudy).sum(axis=(1, 2)))

            in_cloud = in_class & cloudy
            if not in_cloud.any():
                continue
            liquid_here = np.zeros_like(in_cloud)
            for pi, phase in enumerate(PHASE_ORDER_ACC):
                sel = in_cloud & phases[phase]
                if not sel.any():
                    continue
                s_sel = si_grid[sel]
                w_sel = w[sel]
                for scale, edges in edge_sets.items():
                    n_b = n_bars(edges)
                    flat = s_sel * n_b + bar_of[scale][sel]
                    hist[scale][:, code, pi] += np.bincount(
                        flat, weights=w_sel, minlength=n_season * n_b
                    ).reshape(n_season, n_b)
                if phase in PHASE_STACK:
                    liquid_here |= sel

            if liquid_here.any():
                s_sel = si_grid[liquid_here]
                flat = s_sel * n_qbar + qbar[liquid_here]
                qhist[:, code] += np.bincount(
                    flat, weights=w[liquid_here],
                    minlength=n_season * n_qbar,
                ).reshape(n_season, n_qbar)

    return {
        "hist": hist,
        "qhist": qhist,
        "w_class": w_class,
        "w_cloudy": w_cloudy,
        "w_domain": w_domain,
        "w_steps": w_steps,
        "slots": slots,
        "seasons": uniq_seasons,
        "site_code": site_code,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "site_class_counts": site_class_counts,
        "n_unclassified": n_unclassified,
    }


def to_hours_per_season(sec: dict, keep_idx: list[int], edge_sets: dict) -> dict:
    """Convert accumulated weights to hours per season and average the seasons.

    See the module docstring for the normalisation. In one line: divide by the
    class's own cell-hour total so the answer is per cell rather than per class,
    then multiply by the nominal length of the season window so a partially
    sampled season is scaled up rather than counted short.
    """
    n_slot = len(sec["slots"])
    season_hours = n_slot * 24.0 * HOURS_PER_STEP

    w_class = sec["w_class"][keep_idx]                     # (season, class)
    denom = np.where(w_class > 0, w_class, np.nan)

    hours = {}
    for scale in edge_sets:
        h = sec["hist"][scale][keep_idx]                   # (s, class, phase, bar)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_season = h / denom[:, :, None, None] * season_hours
        hours[scale] = {
            "per_season": per_season,                      # kept for the spread
            "mean": nanmean_quiet(per_season, axis=0),     # (class, phase, bar)
        }

    with np.errstate(invalid="ignore", divide="ignore"):
        cloudy_hours = sec["w_cloudy"][keep_idx] / denom * season_hours
        area_pct = 100.0 * w_class / np.where(
            sec["w_domain"][keep_idx][:, None] > 0,
            sec["w_domain"][keep_idx][:, None], np.nan)

    # Quantile histogram: pool the seasons by summing raw weights, so the median
    # is a census over every liquid-bearing cell-hour rather than an average of
    # per-season medians (the median of a pool is not the mean of the medians).
    # This weights a season by how many hours of it the archive actually holds,
    # unlike the bars above, which give every season equal weight. With seasons
    # at 95-100% coverage the two agree closely; they would not for a season
    # half missing, which --min-season-coverage is there to exclude.
    q_pooled = sec["qhist"][keep_idx].sum(axis=0)          # (class, qbar)
    median_lwp_g = np.array([
        weighted_median_from_bins(q_pooled[c], QUANTILE_EDGES_G)
        for c in range(q_pooled.shape[0])
    ])

    return {
        "hours": hours,
        "n_seasons": len(keep_idx),
        "season_hours": season_hours,
        "cloudy_hours": nanmean_quiet(cloudy_hours, axis=0),
        "cloudy_hours_per_season": cloudy_hours,
        "area_pct": nanmean_quiet(area_pct, axis=0),
        "median_lwp_g": median_lwp_g,
        "site_code": sec["site_code"],
    }


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def panel_order(site_code: int) -> list[tuple[int, str, bool]]:
    """(class axis index, label, is_site) for every panel, in drawing order.

    The ARM site is flagged rather than coloured because it is not a sixth
    class -- it is one cell already counted inside whichever class it falls in,
    carried along so ERA5 can be read against the ground observations there.
    """
    out = [(CLASS_CODES[n], CLASS_LABELS[n], False) for n in CLASS_ORDER]
    out.append((site_code, SITE_LABEL, True))
    return out


def make_figure(col: dict, scale: str, edges_g: np.ndarray, region: str,
                mode_label: str, args, output_path: Path | None = None,
                dpi: int | None = None):
    """Draw the six-panel histogram for one bin scale, and save it if asked.

    ``output_path`` of None draws without writing, which is what a notebook
    wants; the figure is returned either way.
    """
    import matplotlib.pyplot as plt

    log_spaced = scale == "log"
    mean_hours = col["hours"][scale]["mean"]        # (class, phase, bar)
    n_bar = mean_hours.shape[2]
    # Drop the underflow bar when nothing is in it, rather than reserving space
    # for a bar that cannot fill. It is empty by construction on the linear
    # scale (edges start at 0) and on the log scale at the default --lwp-log-min
    # (which equals the liquid-bearing floor), but not if either is overridden,
    # so the test is on the data and not on the scale. The OVERFLOW bar is
    # always drawn: an empty one is the evidence that nothing was truncated.
    stack_i = [PHASE_ORDER_ACC.index(p) for p in PHASE_STACK]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Only the DRAWN phases count. The ice-only population is accumulated
        # too and sits in the low bars by construction (an ice cloud's LWP is
        # below --lwp-max-ice), so including it here would keep an underflow bar
        # alive to hold hours that are never plotted in it.
        totals_by_bar = np.nansum(mean_hours[:, stack_i], axis=(0, 1))  # (bar,)
    has_underflow = bool(totals_by_bar[0] > 0)
    first_bar = 0 if has_underflow else 1
    x = np.arange(first_bar, n_bar)

    panels = panel_order(col["site_code"])
    n_r, n_c = args.layout
    if n_r * n_c < len(panels):
        n_c = -(-len(panels) // n_r)
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.6 * n_c, 3.9 * n_r),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    ticks, labels = bar_ticks(edges_g, log_spaced, has_underflow)
    phase_idx = {p: PHASE_ORDER_ACC.index(p) for p in PHASE_STACK}
    pk = col["phase_kw"]
    partitions = pk["iwp_max_liquid_g"] >= pk["iwp_min_g"]

    for k, (code, label, is_site) in enumerate(panels):
        ax = axes[k]
        # Axis furniture first, so an absent class still gets the same axis as
        # its neighbours rather than falling back to matplotlib's default ticks.
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=9.5,
                           rotation=45 if log_spaced else 0,
                           ha="right" if log_spaced else "center")
        # sharex hides the upper rows' labels, which leaves the lower row's
        # titles floating under the upper bars and reading as their x labels.
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_xlim(first_bar - 0.7, n_bar - 0.3)
        if args.y_scale == "log":
            ax.set_yscale("log")

        # A class with no cell-hours at all is a property of the region, not a
        # failure: the Barrow strip, for instance, holds no cell that is pure
        # land by lsm, so every cell touching the coast is "coastal". Say that
        # on the panel rather than drawing an empty axis that reads as a bug.
        if not np.isfinite(col["cloudy_hours"][code]):
            ax.set_title(f"{label}   (absent from this domain)",
                         fontsize=10.5, pad=6, color="#777777")
            ax.text(0.5, 0.5, "no cell of this class\nanywhere in the region",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="#777777")
            continue

        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        bottom = np.zeros(x.size)
        totals = {}
        for phase in PHASE_STACK:
            y = np.nan_to_num(mean_hours[code, phase_idx[phase], first_bar:])
            totals[phase] = float(y.sum())
            ax.bar(x, y, bottom=bottom, width=0.92,
                   color=PHASE_COLORS[phase], edgecolor="none",
                   label=f"{totals[phase]:,.0f} h  {PHASE_LABELS[phase].lower()}")
            bottom = bottom + y

        share = col["area_pct"][code]
        total_h = sum(totals.values())
        pct_season = 100.0 * total_h / col["season_hours"]
        where = "1 cell, inside another class" if is_site else f"{share:.1f}% of area"
        # Only the partitioning case may call the stack "liquid-bearing". With
        # --iwp-max-liquid below --iwp-min the segments leave a gap, and the sum
        # of what is DRAWN is then less than the liquid-bearing hours.
        stack_name = "liquid-bearing" if partitions else "shown"
        ax.set_title(f"{label}   ({where})\n"
                     f"{total_h:,.0f} h {stack_name} = "
                     f"{pct_season:.1f}% of the season",
                     fontsize=10.5, pad=6,
                     fontweight="bold" if is_site else "normal",
                     color=SITE_COLOR if is_site else "black")

        med = col["median_lwp_g"][code]
        xm = value_to_bar_x(med, edges_g, log_spaced)
        handles, hlabels = ax.get_legend_handles_labels()
        if xm is not None and xm >= first_bar - 0.5:
            ax.axvline(xm, color="#B2182B", lw=1.5, ls=":", zorder=6)
            hlabels = hlabels + [f"median LWP = {med:,.3g} g m$^{{-2}}$"]
            handles = handles + [plt.Line2D([], [], color="#B2182B", lw=1.5,
                                            ls=":")]
        if share < args.min_class_area and not is_site:
            hlabels = hlabels + [f"only {share:.2f}% of the domain"]
            handles = handles + [plt.Line2D([], [], color="none")]
        ax.legend(handles, hlabels, fontsize=8.5, framealpha=0.85, loc="best")

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for k in range(0, len(axes), n_c):
        if axes[k].get_visible():
            axes[k].set_ylabel("Hours per season\nper grid cell", fontsize=11)
    for k in range(len(axes) - n_c, len(axes)):
        if axes[k].get_visible():
            axes[k].set_xlabel("LWP [g m$^{-2}$]", fontsize=11)

    degenerate = phase_split_warning(col)
    bin_note = ("log-spaced bins" if log_spaced else
                f"linear bins, {edges_g[1] - edges_g[0]:g} g m$^{{-2}}$ wide")
    fig.suptitle(
        f"Liquid-bearing cloud hours by LWP and surface class — {region}\n"
        f"{mode_label}{', mean across seasons' if col['n_seasons'] > 1 else ''}"
        f"   |   "
        f"season = {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
        f"{args.season_end[0]:02d}-{args.season_end[1]:02d} "
        f"({col['season_hours']:,.0f} h)   |   "
        f"cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}   |   "
        f"liquid-bearing: LWP > {pk['lwp_min_g']:g}, split at IWP = "
        f"{pk['iwp_min_g']:g} g m$^{{-2}}$   |   {bin_note}"
        + ("" if partitions else
           f"   |   !! IWP {pk['iwp_max_liquid_g']:g}-{pk['iwp_min_g']:g} "
           f"g m$^{{-2}}$ falls in neither segment and is not drawn")
        + ("" if degenerate is None else f"\n!!  {degenerate}"),
        fontsize=11.5,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {output_path}")
    return fig


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def add_phase_args(parser: argparse.ArgumentParser) -> None:
    """Cloud phase thresholds, all in g m-2.

    Same four flag names and the same meaning as
    ``cloud_classification.add_cloud_phase_args``, because the masks are built
    by that module's ``cloud_phase_masks``. The DEFAULTS differ, for the two
    reasons set out at DEFAULT_LWP_MIN_G above; they are defined here rather
    than by calling the shared adder so the ``--help`` text states the value
    that is actually used.
    """
    group = parser.add_argument_group("cloud phase (all thresholds in g m-2)")
    group.add_argument(
        "--lwp-min", type=float, default=DEFAULT_LWP_MIN_G, metavar="G",
        help=f"LWP above which a cloud counts as liquid-bearing (default "
             f"{DEFAULT_LWP_MIN_G:g}). Sets the full height of every bar. "
             f"cloud_classification's trace default of 0.03 makes almost every "
             f"overcast ERA5 column qualify.",
    )
    group.add_argument(
        "--iwp-min", type=float, default=DEFAULT_IWP_MIN_G, metavar="G",
        help=f"IWP above which a liquid-bearing cloud counts as MIXED PHASE "
             f"rather than liquid only (default {DEFAULT_IWP_MIN_G:g}). This "
             f"is the split between the two stacked colours.",
    )
    group.add_argument(
        "--iwp-max-liquid", type=float, default=None, metavar="G",
        help="IWP below which a liquid-bearing cloud counts as liquid ONLY "
             "(default: equal to --iwp-min, so the two colours partition the "
             "bar). Set it lower to reopen cloud_classification's gap; the "
             "hours that then fall between the two are reported, not drawn.",
    )
    group.add_argument(
        "--lwp-max-ice", type=float, default=DEFAULT_LWP_MAX_ICE_G, metavar="G",
        help=f"LWP below which a cloudy scene counts as ice only (default "
             f"{DEFAULT_LWP_MAX_ICE_G:g}). Affects the REPORT only: an ice "
             f"cloud has no liquid water path to bin and is never drawn.",
    )


def parse_layout(text: str) -> tuple[int, int]:
    try:
        r, c = str(text).lower().split("x")
        rows, cols = int(r), int(c)
        if rows < 1 or cols < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a panel grid like 2x3") from None
    return rows, cols


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    add_classification_args(parser)
    add_phase_args(parser)
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD", help="Season start (default 08-01).")
    parser.add_argument("--season-end", type=parse_month_day, default=(3, 31),
                        metavar="MM-DD",
                        help="Season end, inclusive (default 03-31, wrapping "
                             "the year).")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC",
                        help="Seasons to average, by the year each season STARTS "
                             "in: '2019', '2019-2025', or '2000,2019-2020'. "
                             "Default: every season meeting "
                             "--min-season-coverage.")
    parser.add_argument("--min-cloud-fraction", type=float,
                        default=DEFAULT_MIN_CLOUD_FRACTION, metavar="F",
                        help="Total cloud cover at or above which a scene counts "
                             f"as cloudy (default {DEFAULT_MIN_CLOUD_FRACTION:g}, "
                             "fully overcast). 0 drops the gate entirely and "
                             "classifies on condensate alone.")
    parser.add_argument("--min-season-coverage", type=float, default=0.6,
                        metavar="F",
                        help="Exclude seasons covering less than this fraction "
                             "of the window (default 0.6). An explicit --years "
                             "is honoured anyway, with a warning.")
    parser.add_argument("--lwp-lin-max", type=float,
                        default=DEFAULT_LWP_LIN_MAX_G, metavar="G",
                        help="Top of the linear LWP axis, g m-2 (default "
                             f"{DEFAULT_LWP_LIN_MAX_G:g}). Anything above it "
                             "goes in the overflow bar.")
    parser.add_argument("--lwp-lin-bins", type=int,
                        default=DEFAULT_LWP_LIN_BINS, metavar="N",
                        help=f"Linear bins from 0 to --lwp-lin-max (default "
                             f"{DEFAULT_LWP_LIN_BINS}, i.e. 25 g m-2 wide).")
    parser.add_argument("--lwp-log-min", type=float,
                        default=DEFAULT_LWP_LOG_MIN_G, metavar="G",
                        help="First log edge, g m-2 (default: equal to "
                             "--lwp-min, below which there are no "
                             "liquid-bearing hours to draw). Set it lower to "
                             "widen the axis; an underflow bar appears only if "
                             "hours actually fall below it.")
    parser.add_argument("--lwp-log-max", type=float,
                        default=DEFAULT_LWP_LOG_MAX_G, metavar="G",
                        help=f"Last log edge, g m-2 (default "
                             f"{DEFAULT_LWP_LOG_MAX_G:g}).")
    parser.add_argument("--lwp-log-bins", type=int,
                        default=DEFAULT_LWP_LOG_BINS, metavar="N",
                        help=f"Log bins between the two edges (default "
                             f"{DEFAULT_LWP_LOG_BINS}; over the default "
                             f"5-1000 g m-2 span that is about 5 per decade).")
    parser.add_argument("--bin-scale", choices=("linear", "log", "both"),
                        default="both",
                        help="Which copies of the figure to draw (default both: "
                             "the same hours binned two ways).")
    parser.add_argument("--y-scale", choices=("linear", "log"), default="linear",
                        help="Scale of the HOURS axis (default linear). This is "
                             "independent of --bin-scale, which sets the LWP "
                             "axis.")
    parser.add_argument("--layout", type=parse_layout, default=DEFAULT_LAYOUT,
                        metavar="RxC",
                        help=f"Panel grid (default "
                             f"{DEFAULT_LAYOUT[0]}x{DEFAULT_LAYOUT[1]} for the "
                             "five classes plus the ARM site). Widened "
                             "automatically if too small.")
    parser.add_argument("--min-class-area", type=float,
                        default=DEFAULT_MIN_CLASS_AREA_PCT, metavar="PCT",
                        help="Stamp a warning on a panel whose class holds less "
                             f"than this %% of the domain (default "
                             f"{DEFAULT_MIN_CLASS_AREA_PCT:g}).")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-figures", action="store_true",
                        help="Print the tables only.")
    return parser.parse_args(argv)


class Analysis(SimpleNamespace):
    """Everything the figures need, computed once.

    Built by :func:`prepare`. ``build_histograms`` accumulates BOTH bin scales
    and every phase in one streaming pass, so a notebook can load once and then
    redraw either copy of the figure for free.
    """


def prepare(argv=None, args=None, **overrides) -> Analysis:
    """Load the archive, classify surface and phase, and bin by LWP.

    ``argv`` takes the same strings as the command line; ``overrides`` sets
    individual options by name, e.g. ``prepare(region="barrow",
    years=(2019, 2020))``. Returns an :class:`Analysis`.

    This is the slow step. ``--bin-scale`` is deliberately NOT honoured here:
    both scales are accumulated regardless, so the notebook's figure cells are
    all available without a reload.
    """
    if args is None:
        args = parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    print("=" * 72)
    print("Liquid-bearing cloud hours by LWP and surface class")
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

    phase_kw = resolve_phase_thresholds(args)
    log_min = (phase_kw["lwp_min_g"] if args.lwp_log_min is None
               else args.lwp_log_min)
    edge_sets = {
        "linear": linear_bin_edges(args.lwp_lin_max, args.lwp_lin_bins),
        "log": log_bin_edges(log_min, args.lwp_log_max, args.lwp_log_bins),
    }

    print(f"  Source     : {region_dir}")
    print(f"  Grid       : {ds.sizes['latitude']} x {ds.sizes['longitude']} "
          f"cells, {ds.sizes['valid_time']:,} time steps")
    print(f"  Season     : {args.season_start[0]:02d}-{args.season_start[1]:02d}"
          f" to {args.season_end[0]:02d}-{args.season_end[1]:02d}"
          + ("  (wraps the new year)"
             if args.season_end < args.season_start else ""))
    print(f"  Classes    : lsm tol {args.lsm_tol:g} | open ocean < "
          f"{args.open_ocean_max_siconc:g} | pack ice > "
          f"{args.sea_ice_min_siconc:g}")
    print(f"  Cloudy     : tcc >= {args.min_cloud_fraction:g}")
    print(f"  Phase      : liquid-bearing = LWP > {phase_kw['lwp_min_g']:g} | "
          f"liquid only = IWP < {phase_kw['iwp_max_liquid_g']:g} | "
          f"mixed = IWP > {phase_kw['iwp_min_g']:g}  (g m-2)")
    if phase_kw["iwp_max_liquid_g"] < phase_kw["iwp_min_g"]:
        print("               !! --iwp-max-liquid is below --iwp-min, so "
              "liquid-bearing hours with IWP between them belong to neither "
              "segment and are not drawn.")
    print(f"  LWP bins   : linear 0-{args.lwp_lin_max:g} in "
          f"{args.lwp_lin_bins} | log {log_min:g}-"
          f"{args.lwp_log_max:g} in {args.lwp_log_bins}")

    layout = season_layout(ds, args)
    keep_idx, used, mode_label = select_seasons(layout, args)
    print(f"\n  Reading {len(used)} season(s): {used}")

    sec = build_histograms(ds, lsm, args, layout, keep_idx, edge_sets, phase_kw)
    if sec["n_unclassified"]:
        print(f"  !! {sec['n_unclassified']:,} unclassified cell-times; run "
              f"surface_classification.py for the breakdown.", file=sys.stderr)
    col = to_hours_per_season(sec, keep_idx, edge_sets)
    col["phase_kw"] = phase_kw

    tag = f"season{used[0]}" if len(used) == 1 else f"mean{used[0]}-{used[-1]}"
    return Analysis(args=args, ds=ds, lsm=lsm, layout=layout, keep_idx=keep_idx,
                    used=used, mode_label=mode_label, sec=sec, col=col,
                    edge_sets=edge_sets, phase_kw=phase_kw, tag=tag)


def print_report(A: Analysis) -> None:
    """Hours per season per class and phase, and where the ARM site cell landed.

    Prints the ice-only and unclassified-phase populations too. They are not on
    the figure -- an ice cloud has no liquid water path to bin -- but how much of
    a class's cloudy time they account for is the context the figure lacks.
    """
    col, sec = A.col, A.sec
    mean_lin = col["hours"]["linear"]["mean"]        # (class, phase, bar)
    phase_i = {p: PHASE_ORDER_ACC.index(p) for p in PHASE_ORDER_ACC}
    season_h = col["season_hours"]

    print(f"\n  Hours per season per grid cell ({season_h:,.0f} h in the window),"
          f" mean over {len(A.used)} season(s):")
    print(f"    {'class':<22}{'area %':>8}{'cloudy':>10}{'liquid':>10}"
          f"{'mixed':>10}{'ice':>10}{'liq+mix %':>11}{'med LWP':>10}")
    print("    " + "-" * 91)
    for code, label, _ in panel_order(col["site_code"]):
        liq = float(np.nansum(mean_lin[code, phase_i["liquid"]]))
        mix = float(np.nansum(mean_lin[code, phase_i["mixed"]]))
        ice = float(np.nansum(mean_lin[code, phase_i["ice"]]))
        cloudy = col["cloudy_hours"][code]
        pct = 100.0 * (liq + mix) / cloudy if cloudy > 0 else float("nan")
        area = col["area_pct"][code]
        area_s = "  1 cell" if code == col["site_code"] else f"{area:8.2f}"
        print(f"    {label:<22}{area_s}{cloudy:>10.0f}{liq:>10.0f}{mix:>10.0f}"
              f"{ice:>10.0f}{pct:>11.1f}{col['median_lwp_g'][code]:>10.3g}")
    print("    " + "-" * 91)
    pk = col["phase_kw"]
    print("    liquid + mixed is the bar height drawn; 'cloudy' is every "
          "overcast hour, which also")
    print("    holds the ice-only scenes and the ones whose LWP falls "
          "between --lwp-max-ice")
    print(f"    ({pk['lwp_max_ice_g']:g}) and --lwp-min ({pk['lwp_min_g']:g} "
          f"g m-2) and so belong to no phase at all.")
    if pk["iwp_max_liquid_g"] < pk["iwp_min_g"]:
        print(f"    !! --iwp-max-liquid ({pk['iwp_max_liquid_g']:g}) is below "
              f"--iwp-min ({pk['iwp_min_g']:g}), so liquid-bearing")
        print("    hours with IWP between them are in NEITHER column and "
              "are not drawn.", file=sys.stderr)

    degenerate = phase_split_warning(col)
    if degenerate:
        print(f"\n  !! {degenerate}.", file=sys.stderr)

    # The scale-up the module docstring warns about, made visible: how many
    # hours each season actually contributed against the window they are
    # reported over.
    sampled = sec["w_steps"][A.keep_idx]
    if sampled.size:
        print(f"\n  Hours present in the archive per season: "
              f"{sampled.min():,.0f} to {sampled.max():,.0f} of {season_h:,.0f}."
              f"  Bar heights are rates scaled to the full window.")

    if len(A.used) > 1:
        print("\n  Spread across seasons, liquid-bearing hours per season:")
        per = A.col["hours"]["linear"]["per_season"]      # (s, class, phase, bar)
        stack_i = [phase_i[p] for p in PHASE_STACK]
        tot = np.nansum(per[:, :, stack_i, :], axis=(2, 3))   # (season, class)
        for code, label, _ in panel_order(col["site_code"]):
            v = tot[:, code]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            print(f"    {label:<22}min {v.min():7.0f}   median "
                  f"{np.median(v):7.0f}   max {v.max():7.0f}")

    d_lat = sec["site_lat"] - SITE_LAT
    d_lon = sec["site_lon"] - SITE_LON
    print(f"\n  {SITE_LABEL}: facility at {SITE_LAT:.3f} N, {SITE_LON:.3f} E")
    print(f"    nearest cell centre  : {sec['site_lat']:.3f} N, "
          f"{sec['site_lon']:.3f} E  (offset {d_lat:+.3f}, {d_lon:+.3f} deg)")
    counts = sec["site_class_counts"]
    total = int(counts.sum())
    if total:
        print("    the cell classifies as, over the window:")
        for name in CLASS_ORDER:
            n = int(counts[CLASS_CODES[name]])
            if n:
                print(f"      {CLASS_LABELS[name]:<20}{100 * n / total:6.2f}%")
        if counts[-1]:
            print(f"      {'Unclassified':<20}{100 * counts[-1] / total:6.2f}%")


def figure(A: Analysis, scale: str, out_dir=None, dpi: int | None = None):
    """Draw one bin scale's six-panel figure from a prepared :class:`Analysis`."""
    if scale not in A.edge_sets:
        raise KeyError(f"unknown bin scale {scale!r}; "
                       f"choose from {sorted(A.edge_sets)}")
    path = None
    if out_dir is not None:
        suffix = "_logy" if A.args.y_scale == "log" else ""
        path = (Path(out_dir) /
                f"{A.args.region}_lwp_hist_surfaceclass_{scale}{suffix}_"
                f"{A.tag}.png")
    return make_figure(A.col, scale, A.edge_sets[scale], A.args.region,
                       A.mode_label, A.args, path, dpi)


def fig_linear(A: Analysis, out_dir=None, dpi: int | None = None):
    """Linear LWP bins -- the physical-axis copy."""
    return figure(A, "linear", out_dir, dpi)


def fig_log(A: Analysis, out_dir=None, dpi: int | None = None):
    """Log-spaced LWP bins -- the copy that resolves the low-LWP end."""
    return figure(A, "log", out_dir, dpi)


ALL_FIGURES = (fig_linear, fig_log)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scales = ["linear", "log"] if args.bin_scale == "both" else [args.bin_scale]
    try:
        A = prepare(args=args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"  Error: {exc}", file=sys.stderr)
        return 1
    print_report(A)
    if args.no_figures:
        print("=" * 72)
        return 0

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "figures")
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    print()
    for scale in scales:
        figure(A, scale, out_dir=out_dir)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
