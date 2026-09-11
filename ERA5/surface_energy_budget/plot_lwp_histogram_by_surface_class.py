#!/usr/bin/env python3
"""Hours per season with a liquid-bearing cloud, binned by LWP, per surface class.

One panel per surface class from ``surface_classification.py`` -- land, coastal,
open ocean, marginal ice, sea ice -- plus a sixth panel for the single ERA5 grid
cell holding the DOE ARM facility at Utqiagvik. Inside each panel the bars are
stacked by cloud phase from ``cloud_classification.py``: liquid-only clouds at
the bottom, mixed-phase on top. The two are defined by four INDEPENDENT
thresholds (see CLOUD PHASE) and are guaranteed not to overlap, so the bar
height is exactly their sum -- and, deliberately, not every cloud is in one.

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
    log      log-spaced bins, by default from the lower of the two LWP floors
             (below which nothing is drawn at all) to --lwp-log-max.
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
The two drawn categories have INDEPENDENT thresholds -- four numbers, not two --
because they answer different questions. Masks come from
``cloud_classification.liquid_mixed_masks``, applied to scenes that already pass
the cloud-cover test:

    cloudy        tcc >= --min-cloud-fraction        (default 1.0, overcast)
    liquid only   cloudy and LWP > --liquid-lwp-min  (5) and IWP < --liquid-iwp-max (1)
    mixed phase   cloudy and LWP > --mixed-lwp-min   (1) and IWP > --mixed-iwp-min  (1)
    ice only      cloudy and IWP > --mixed-iwp-min   and LWP < --lwp-max-ice (0.001)
    neither       cloudy and none of the above

"Liquid only" is a claim about a deck that is radiatively liquid: it wants a
SUBSTANTIAL liquid path and ice near enough to absent to be negligible. "Mixed
phase" only claims both species are present, so a low floor on each is right.
Tying them to one shared floor, as ``cloud_phase_masks`` does, makes one of the
two answer the wrong question.

THE ONE CONSTRAINT BETWEEN THEM
-------------------------------
    --liquid-iwp-max <= --mixed-iwp-min

and nothing else. Writing the categories as sets,

    liquid only = {LWP > a} and {IWP < b}
    mixed phase = {LWP > c} and {IWP > d}

their intersection is ``{LWP > max(a, c)} and {d < IWP < b}``. LWP is unbounded
above, so the first factor is NEVER empty: the LWP floors cannot separate the
categories however far apart they are set, and disjointness rests entirely on
the IWP axis. VERIFIED by exhaustive sweep in
``cloud_classification.check_liquid_mixed_disjoint``, which raises rather than
let a scene be counted twice. Equality is allowed and is the tightest useful
setting.

NOT EXHAUSTIVE, AND THE REMAINDER IS REPORTED
---------------------------------------------
The two categories deliberately do not cover every cloud. At the defaults a
scene with LWP 3 and IWP 0.5 g m-2 is too thin for "liquid only" and too dry for
"mixed phase", and belongs to neither. That population is real cloud, not error,
and it is large: over the Barrow strip in Oct-Mar it runs 60-655 hours a season
depending on the surface class. The report gives it a column of its own, and
liquid + mixed + ice + neither is CHECKED to reconstruct the cloudy hours
exactly, with the residual printed.

Because of that, a bar's full height is "liquid + mixed" and the panel title
says exactly that. It is NOT "all liquid-bearing cloud", and calling it that
would quietly hide the remainder.

Ice-only scenes are NOT drawn. An ice cloud has no liquid water path to bin, so
every one of them would pile into the first bar and say nothing about LWP. They
are in the text summary instead, because how much of a class's cloudy time is
ice-only is exactly the context the figure lacks.

WHY THE THRESHOLD DEFAULTS ARE NOT cloud_classification's
---------------------------------------------------------
``cloud_classification.cloud_phase_masks`` defaults its floors to 0.03 g m-2,
about ERA5's trace quantum (see ERA5_TYPICAL_PATH_QUANTUM_G below -- the quantum
is per-field and runs 0.015 to 0.061 g m-2 across this archive, so "the" trace
level is a range, not a number). At that level essentially every overcast Arctic
column holds a trace of both species, so "mixed phase" degenerates into
"overcast" and the stack collapses to one colour. MEASURED, Barrow strip,
November 2022, tcc >= 0.99: of the overcast columns with LWP > 0.031 g m-2,
99.92% carry at least two quanta of ice, leaving "liquid only" with 0.08% of the
drawn hours -- one or two hours a season. ``phase_split_warning`` now catches
that at run time and says so on the figure. Raise --liquid-lwp-min to 10-25 to
sit near a microwave radiometer's detection floor when comparing against a
ground-based retrieval.

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
--------------------------
--liquid-lwp-min G           Liquid-only LWP floor (default 5).
--liquid-iwp-max G           Liquid-only IWP ceiling (default 1).
--mixed-lwp-min G            Mixed-phase LWP floor (default 1).
--mixed-iwp-min G            Mixed-phase IWP floor (default 1).
                             Requires --liquid-iwp-max <= --mixed-iwp-min.
--lwp-max-ice G              Ice-only ceiling (default 0.001). Report only.
--min-cloud-fraction F       Cloud cover for a cloudy scene (default 1.0).

Binning
-------
--lwp-lin-max G              Top of the linear axis (default 600).
--lwp-lin-bins N             Linear bins below it (default 24, i.e. 25 g m-2).
--lwp-log-min G              First log edge (default: the lower LWP floor).
--lwp-log-max G              Last log edge (default 1000).
--lwp-log-bins N             Log bins between them (default 12).
--bin-scale {linear,log,both}  Which copies to write (default both).
--y-scale {linear,log}       Scale of the hours axis (default linear).

Examples
--------
    ./plot_lwp_histogram_by_surface_class.py --region barrow --years 2015-2025

    ./plot_lwp_histogram_by_surface_class.py --region barrow --years 2015-2025 \
        --season-start 11-01 --season-end 02-28 --liquid-lwp-min 25

Requires the mask from ``download_era5_land_sea_mask.py`` for the region.
"""

from __future__ import annotations

import argparse
import calendar
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cloud_classification import (
    DEFAULT_ICE_FRACTION_MIN,
    DEFAULT_LIQUID_FRACTION_MIN,
    DEFAULT_LIQUID_IWP_MAX_G,
    DEFAULT_LIQUID_LWP_MIN_G,
    DEFAULT_LWP_MAX_ICE_G,
    DEFAULT_MIN_IWP_G,
    DEFAULT_MIN_LWP_G,
    DEFAULT_MIXED_IWP_MIN_G,
    DEFAULT_MIXED_LWP_MIN_G,
    PHASE_COLORS,
    PHASE_LABELS,
    check_fraction_thresholds_disjoint,
    check_liquid_mixed_disjoint,
    fraction_phase_masks,
    liquid_mixed_masks,
)
from seb_analysis_common import (
    add_data_source_args,
    load_seb_data,
    resolve_data_root,
    resolve_region_dir,
)
from surface_classification import (
    CLASS_CODES,
    CLASS_COLORS,
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
    SITE_KEY,
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

# ----------------------------------------------------------------------------
# LEAP YEARS: the season window is not one length
# ----------------------------------------------------------------------------
# season_calendar() builds its day-slot list on a LEAP reference year so that
# 29 February always has a column. That is the right choice for indexing -- one
# slot axis shared by every season -- but it is the wrong denominator, and using
# it as one produced two errors that both looked like missing data:
#
#   * COVERAGE. A complete Oct-Mar season of a common year has 182 of the 183
#     slots filled, so it reported 99.5% and looked slightly gappy. Only seasons
#     ending in a leap year (2015/16, 2019/20, 2023/24 here) reached 100%.
#
#   * SCALING. to_hours_per_season scales each season up to the nominal window,
#     so a complete common year was multiplied by 183/182 -- credited with 24
#     hours it never had. The monthly form was worse: February took 29 days for
#     every season, inflating every common-year February by 1/28 = 3.6%.
#
# Both denominators are now per season, from layout["days_per_season"] and
# layout["month_days_per_season"]. The property to preserve when touching this:
# A COMPLETE SEASON MUST SCALE BY EXACTLY 1.0, whether or not it holds 29
# February. Scaling is for genuinely missing files -- 2017/18 at 93.4% -- and
# must be a no-op for a season that is simply shorter.
#
# Consequence for callers: season_phase_hours returns season_h as an ARRAY over
# seasons, not a float. Use _sh(season_h, i) for one season and window_label()
# to render it.

REQUIRED_VARS = ("tcc", "tclw", "tciw", "siconc")

# ----------------------------------------------------------------------------
# Which ERA5 field stands for "liquid"
# ----------------------------------------------------------------------------
# tclw  total column CLOUD liquid water -- all suspended cloud liquid, at any
#       temperature, excluding precipitating rain and drizzle.
# tcslw total column SUPERCOOLED liquid water -- liquid existing below 0 C.
#
# Physically the second ought to be a subset of the first. IN THE ARCHIVED
# SINGLE-LEVEL FIELDS IT IS NOT: over the Barrow strip, tcslw exceeds tclw by
# more than the 0.031 g m-2 storage quantum in 22.5% of cells in January 2023
# and 38.5% in November 2022, by up to +361 g m-2, with a p95 ratio near 3.5.
# Measured, not inferred. Three explanations were tested and rejected:
#
#   * precipitating liquid folded in -- tcrw is below the quantum in 99% of the
#     exceeding cells, so rain cannot account for it;
#   * a time-axis offset between the two fields -- the lag-0 correlation
#     (r = 0.938) is higher than lag +/-1 (0.886, 0.898), so they are aligned;
#   * ordinary quantisation -- the excess survives a full-quantum margin.
#
# So swapping one for the other is NOT a strict narrowing, and the difference
# in cloud hours is a SIGNED change: some scenes lose liquid, others gain it.
# Any figure built on this pair has to report both directions rather than
# calling the difference a loss. The cause is a property of ERA5's own
# diagnostics and is not resolved here.
LIQUID_VARS: tuple[str, ...] = ("tclw", "tcslw")
DEFAULT_LIQUID_VAR = "tclw"

LIQUID_VAR_LABEL = {
    "tclw": "cloud liquid water (tclw)",
    "tcslw": "supercooled liquid water (tcslw)",
}
LIQUID_VAR_SHORT = {"tclw": "all liquid", "tcslw": "supercooled"}


def parse_utc_hours(text) -> tuple[int, ...] | None:
    """'7,8,9' or '7-9,19-21' -> the UTC hours to keep; None or '' means all.

    Exists because tclw and tcslw are not the same kind of product: tclw is the
    4D-Var analysis at step 0, while tcslw comes from a free-running forecast
    initialised at 06 and 18 UTC, so its error against the analysis grows with
    the forecast lead. Restricting to short-lead hours is the only way to
    compare the two fields without that error dominating. See
    compare_liquid_definitions.hours_for_lead.
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple, set)):
        vals = sorted({int(v) for v in text})
        return tuple(vals) if vals else None
    text = str(text).strip()
    if not text or text.lower() == "all":
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            lo, hi = sorted((int(a), int(b)))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    bad = sorted(h for h in out if not 0 <= h <= 23)
    if bad:
        raise ValueError(f"--utc-hours must be 0-23; got {bad}")
    if not out:
        return None
    return tuple(sorted(out))


def utc_hour_mask(ds, args) -> np.ndarray:
    """Per-time-step boolean: is this step's UTC hour selected?

    All True when --utc-hours is unset, so callers apply it unconditionally.
    """
    times = np.asarray(ds["valid_time"].values)
    hours = parse_utc_hours(getattr(args, "utc_hours", None))
    if hours is None:
        return np.ones(times.shape[0], dtype=bool)
    h = times.astype("datetime64[h]").astype(np.int64) % 24
    return np.isin(h, np.asarray(hours, dtype=np.int64))

# ----------------------------------------------------------------------------
# Precipitation filter
# ----------------------------------------------------------------------------
# The literature threshold for "ERA5 says it is precipitating" is 0.1 mm/hr,
# chosen to match a rain gauge's minimum measurable amount. That is a RATE, and
# ERA5 carries it directly as tp. tcrw + tcsw is a suspended MASS PATH, a
# different quantity, so the two are not interchangeable and the mass-path
# threshold below was CALIBRATED against the rate rather than assumed.
#
# The mass-path threshold below matters only for precip_var="path". The default
# is precip_var="rate", which needs no calibration at all.
#
# MEASURED on this archive (40 files spanning the record):
#
#   median rain+snow path where tp is 0.08-0.12 mm/hr      48.8 g m-2
#   physical estimate  W = R*H/v, H = 1 km, rain  v=4 m/s    6.9 g m-2
#                                          snow  v=1 m/s   27.8 g m-2
#
# 50 g m-2 sits at the measured median and is consistent with the physical
# estimate for a 1.5-2 km deep snow layer, which is what an Arctic precipitating
# column looks like. Classification agreement against the rate threshold is
# 93.6% (4.9% false wet, 1.5% false dry).
#
# THE MAPPING IS LOOSE, AND THAT IS A PROPERTY OF THE DATA, NOT THE CHOICE.
# Among cell-hours with tp >= 0.1 mm/hr the rain+snow path spans 21 to 599
# g m-2 between the 5th and 95th percentiles, so NO single mass-path cut
# reproduces the rate cut cleanly. If the comparison needs the literature
# definition exactly, use precip_var="rate", which applies 0.1 mm/hr to tp.
DEFAULT_PRECIP_PATH_MAX_G = 50.0     # g m-2 of tcrw + tcsw
DEFAULT_PRECIP_RATE_MAX_MM_HR = 0.1  # mm hr-1 of tp
PRECIP_VARS: tuple[str, ...] = ("path", "rate")
# tp is the default because it is the quantity the published 0.1 mm/hr
# convention is actually defined on, and it is already in the archive -- present
# in all 654 single-level files, covering every season 2014/15-2025/26 with no
# gaps. The mass-path route below stays available, but it estimates a rate from
# a suspended burden and the two only agree to about 94%; there is no reason to
# accept that error when the rate itself is on disk.
DEFAULT_PRECIP_VAR = "rate"

# Only loaded when the filter is on, so an archive without them still works.
PRECIP_SOURCE_VARS = {"path": ("tcrw", "tcsw"), "rate": ("tp",)}


def precip_mask(block, keep, args) -> np.ndarray:
    """True where the scene is PRECIPITATING and should be filtered out.

    Returns an all-False mask when the filter is off, so the caller can apply
    it unconditionally.
    """
    if not args.no_precip:
        return np.zeros(block["tcc"].values[keep].shape, dtype=bool)

    if args.precip_var == "rate":
        # tp is an hourly accumulation in metres; 1 m over 1 h = 1000 mm/hr.
        rate = block["tp"].values[keep] * 1000.0
        return np.isfinite(rate) & (rate >= args.precip_rate_max)

    path_g = (block["tcrw"].values[keep] + block["tcsw"].values[keep]) * 1000.0
    return np.isfinite(path_g) & (path_g >= args.precip_path_max)


def precip_label(args) -> str:
    """One-line description of the filter, for figure subtitles."""
    if not args.no_precip:
        return "no precipitation filter"
    if args.precip_var == "rate":
        return f"non-precipitating: tp < {args.precip_rate_max:g} mm hr$^{{-1}}$"
    return (f"non-precipitating: rain+snow path < "
            f"{args.precip_path_max:g} g m$^{{-2}}$")



# Phases drawn, in stack order from the bottom. "ice" is accumulated as well but
# never plotted -- it has no liquid water path to bin. See the module docstring.
PHASE_STACK: tuple[str, ...] = ("liquid", "mixed")

# "none" is every cloudy hour matching no category. Accumulating it explicitly,
# rather than inferring it by subtraction, makes liquid + mixed + ice + none ==
# cloudy an INVARIANT the run can assert instead of an assumption -- and with
# four independent thresholds that population is no longer a rounding-level
# curiosity. At the defaults it is every cloud holding 1-5 g m-2 of liquid and
# almost no ice: too thin for "liquid only", too dry for "mixed".
PHASE_ORDER_ACC: tuple[str, ...] = ("liquid", "mixed", "ice", "none")

# Cloud-cover gate, matching plot_surface_class_timeseries.py and
# analyze_cloud_liquid_frequency.py rather than cloud_classification's
# --cloudy-threshold. See the module docstring.
DEFAULT_MIN_CLOUD_FRACTION = 1.0

# Phase thresholds, g m-2. The masks come from
# ``cloud_classification.liquid_mixed_masks``, which takes FOUR independent
# numbers so the two drawn categories can be defined on their own terms:
#
#     liquid only   LWP > --liquid-lwp-min  and  IWP < --liquid-iwp-max
#     mixed phase   LWP > --mixed-lwp-min   and  IWP > --mixed-iwp-min
#
# The two questions are not symmetric, which is why they need separate floors.
# "Liquid only" is a claim about a deck that is radiatively liquid, so it wants
# a substantial liquid path AND ice near enough to absent to be negligible.
# "Mixed phase" only claims both species are present, so a low floor on each is
# right. Forcing them onto one shared floor, as ``cloud_phase_masks`` does,
# makes one of the two answer the wrong question.
#
# NOT the trace defaults. cloud_classification uses 0.03 g m-2, ERA5's
# quantisation scale, at which essentially every overcast Arctic column holds a
# trace of both species: MEASURED, Barrow strip, November 2022, tcc >= 0.99,
# 99.92% of columns with LWP > 0.031 carry at least two quanta of ice, so
# "liquid only" collapses to one or two hours a season and the stack becomes a
# single colour. See phase_split_warning, which now catches that at runtime.
DEFAULT_LIQUID_LWP_MIN = DEFAULT_LIQUID_LWP_MIN_G    # 5.0
DEFAULT_LIQUID_IWP_MAX = DEFAULT_LIQUID_IWP_MAX_G    # 1.0
DEFAULT_MIXED_LWP_MIN = DEFAULT_MIXED_LWP_MIN_G      # 1.0
DEFAULT_MIXED_IWP_MIN = DEFAULT_MIXED_IWP_MIN_G      # 1.0

# Ice-only gets its own IWP floor rather than borrowing the mixed-phase one.
# "How much ice makes a cloud an ice cloud" and "how much ice makes a liquid
# cloud mixed" are separate judgements, and nothing forces them to agree. The
# default keeps them equal, so behaviour is unchanged unless it is set.
DEFAULT_ICE_IWP_MIN = DEFAULT_MIXED_IWP_MIN_G        # 1.0

# ---------------------------------------------------------------------------
# The FRACTION scheme (--phase-mode fraction)
# ---------------------------------------------------------------------------
# Classify on each species' share of the cloud water path CWP = LWP + IWP
# instead of on absolute magnitudes:
#
#     liquid only   LWP/CWP >= --liquid-fraction-min   (default 0.90)
#     ice only      IWP/CWP >= --ice-fraction-min      (default 0.90)
#     mixed phase   everything else holding cloud water
#
# Exhaustive and disjoint by construction, so there is no "neither" population
# beyond scenes carrying no cloud water at all -- which is the scheme's main
# advantage over the absolute one, and why it needs no gap accounting. The
# minimum paths are what keep the ratio off numerical dust: a scene with
# 0.05 g m-2 of liquid and nothing else is 100% liquid by share, and without a
# floor would be reported as a liquid cloud.
DEFAULT_LIQUID_FRACTION = DEFAULT_LIQUID_FRACTION_MIN   # 0.90
DEFAULT_ICE_FRACTION = DEFAULT_ICE_FRACTION_MIN         # 0.90
DEFAULT_MIN_LWP = DEFAULT_MIN_LWP_G                     # 0.1 g m-2
DEFAULT_MIN_IWP = DEFAULT_MIN_IWP_G                     # 0.1 g m-2

# ---------------------------------------------------------------------------
# The --min-lwp SWEEP (fraction mode only)
# ---------------------------------------------------------------------------
# How much of the phase split is a property of the atmosphere and how much is a
# property of the threshold? The only honest answer is to vary the threshold and
# look. Every value in the sweep is classified in the SAME streaming pass as the
# nominal one, so the sensitivity costs one archive read rather than N of them --
# which, at nine minutes a read, is the difference between a routine check and
# one nobody runs.
#
# Fraction mode only. In absolute mode there is no single "the minimum LWP":
# liquid-only and mixed-phase carry separate floors, and sweeping them together
# would change what the categories mean rather than how sensitive they are.
DEFAULT_SWEEP_LWP_MIN = 0.05      # g m-2
DEFAULT_SWEEP_LWP_MAX = 20.0      # g m-2
DEFAULT_SWEEP_POINTS = 21         # linearly spaced: 0.05, ~1.05, ~2.05, ... 20

# MEASURED: over the Barrow strip the phase split barely moves below about
# 1 g m-2 -- the curves are flat from 0.05 to 1 -- and does nearly all of its
# moving between 1 and 20. Log spacing put two thirds of its points, and two
# thirds of the axis, in the flat part. Linear spacing puts the resolution where
# the answer actually changes. ``--sweep-spacing log`` restores the old
# behaviour, which is still the better choice for looking at the region where
# the threshold is fighting ERA5's own quantisation (below ~0.1 g m-2).
DEFAULT_SWEEP_SPACING = "linear"
SWEEP_SPACINGS: tuple[str, ...] = ("linear", "log")

# Phase labels the sweep accumulates, in the order it stores them. "none" is
# carried so the four still partition the overcast hours and the partition can
# be checked, even though only the first three are drawn.
SWEEP_PHASES: tuple[str, ...] = ("liquid", "ice", "mixed", "none")
SWEEP_DRAWN: tuple[str, ...] = ("liquid", "ice", "mixed")


def sweep_drawn_phases(args: argparse.Namespace) -> tuple[str, ...]:
    """Phases the sweep figures draw, honouring ``--show-ice-only``.

    Off by default: the ice-only curve rises whenever liquid gets zeroed out
    of a mixed-phase scene, not when ice itself increases, which reads as
    "more ice" to anyone who hasn't seen the reclassification argument in the
    notebook. See the 'Reading these figures' section there.
    """
    if getattr(args, "show_ice_only", False):
        return SWEEP_DRAWN
    return tuple(p for p in SWEEP_DRAWN if p != "ice")

# Class axis of the sweep accumulator: the five classes, then UNCLASSIFIED. This
# is a genuine partition of every cell, which is what lets one np.bincount fill
# all six at once; "all cells" is then their sum, exactly, with no assumption
# that unclassified is empty.
N_SWEEP_CLASS = len(CLASS_ORDER) + 1
SWEEP_UNCLASSIFIED_SLOT = len(CLASS_ORDER)

PHASE_MODES: tuple[str, ...] = ("absolute", "fraction")
DEFAULT_PHASE_MODE = "absolute"

# ERA5's condensate paths are quantised, because GRIB stores them with a binary
# scale factor. NOTE 2**-15 is TWO TO THE POWER OF MINUS FIFTEEN -- 1/32768, or
# 3.05e-5 kg m-2 -- not 2e-15; the Python operator is easy to misread as
# scientific notation, and the difference is ten orders of magnitude.
#
# The scale factor is chosen PER FIELD, so the quantum is not one fixed number
# the way cloud_classification.py's note implies. MEASURED over all 641 barrow
# files carrying tciw:
#
#     2**-14 kg m-2 = 0.0610 g m-2      1 file
#     2**-15 kg m-2 = 0.0305 g m-2    617 files
#     2**-16 kg m-2 = 0.0153 g m-2     23 files
#
# so the finest path this archive actually expresses is 0.0153 g m-2, half the
# usually-quoted figure, and a threshold meant to drop "the single-quantum
# population" drops a different number of quanta depending on which file the
# hour came from.
#
# The common value below is the SCALE for the degeneracy test only, where what
# matters is the order of magnitude. An ice threshold within a quantum or two of
# it does not separate "no ice" from "some ice" -- it separates "literally zero
# ice" from everything else, which in an overcast Arctic column essentially
# never happens. MEASURED, Barrow strip, November 2022, tcc >= 0.99,
# LWP > 0.031 g m-2: 99.92% of those cell-hours carry at least two quanta of
# ice, so a 0.031 g m-2 ice threshold leaves "liquid only" holding 0.08% of the
# drawn hours -- one or two hours a season.
ERA5_TYPICAL_PATH_QUANTUM_G = 2.0 ** -15 * 1000.0    # 0.0305176 g m-2

# A stacked segment below this share of the drawn hours has collapsed:
# the figure is one colour and the split conveys nothing. Not an error -- the
# thresholds were honoured exactly -- so it is reported, not raised.
DEGENERATE_SEGMENT_SHARE = 0.01

# Display bin defaults. The linear set reproduces a 0-600 g m-2 axis in
# 25 g m-2 steps; the log set matches analyze_cloud_liquid_frequency.py.
DEFAULT_LWP_LIN_MAX_G = 600.0
DEFAULT_LWP_LIN_BINS = 24
# None means "start the log axis at the lower of the two LWP floors". Below
# that there is nothing to draw by definition, so a fixed 0.1 g m-2 default
# would spend most of the axis on bins that cannot contain anything.
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
def sweep_lwp_values(lo_g: float, hi_g: float, n: int,
                     spacing: str = DEFAULT_SWEEP_SPACING) -> np.ndarray:
    """Minimum-LWP values for the sweep, in g m-2.

    Linear by default -- see DEFAULT_SWEEP_SPACING for the measurement behind
    that choice. ``spacing="log"`` resolves the sub-0.1 g m-2 region instead,
    where the threshold interacts with ERA5's quantisation rather than with the
    cloud.

    A lower bound of exactly 0 is allowed under linear spacing and means "any
    non-zero liquid counts", which is a meaningful left anchor for the axis.
    Log spacing cannot start there.
    """
    if spacing not in SWEEP_SPACINGS:
        raise ValueError(f"unknown sweep spacing {spacing!r}; "
                         f"choose from {list(SWEEP_SPACINGS)}")
    if hi_g <= lo_g or n < 2:
        raise ValueError("need sweep min < sweep max and at least 2 points")
    if spacing == "log":
        if lo_g <= 0:
            raise ValueError("log sweep spacing needs a sweep min above 0; "
                             "use --sweep-spacing linear to start at 0")
        return np.geomspace(float(lo_g), float(hi_g), int(n))
    if lo_g < 0:
        raise ValueError("sweep min cannot be negative")
    return np.linspace(float(lo_g), float(hi_g), int(n))


def season_window_hours(layout: dict, keep_idx) -> np.ndarray:
    """Hours in each SELECTED season's own window, leap years included.

    A common-year Oct-Mar season is 182 days, a leap-year one 183. Scaling
    every season to the longer of the two would credit a common year with 24
    hours it never had -- and, worse, would stop a complete common year from
    passing through the normalisation unchanged.
    """
    days = np.asarray(layout["days_per_season"], dtype=float)[list(keep_idx)]
    return days * 24.0 * HOURS_PER_STEP


def season_month_window_hours(layout: dict, keep_idx) -> np.ndarray:
    """Hours of each calendar month present, per selected season.

    Shaped ``(n_season, n_month)``. February is the reason this cannot be one
    row shared by every season: 28 days in a common year and 29 in a leap year
    is a 3.6% difference in that month's denominator, which is an order of
    magnitude larger than the whole-season effect.
    """
    md = np.asarray(layout["month_days_per_season"], dtype=float)[list(keep_idx)]
    return md * 24.0 * HOURS_PER_STEP


def window_label(season_h) -> str:
    """Render a season-window length that may vary between seasons."""
    a = np.asarray(season_h, dtype=float)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return f"{lo:,.0f} h" if lo == hi else f"{lo:,.0f}-{hi:,.0f} h"


def month_window_hours(slots: list[tuple[int, int]]) -> np.ndarray:
    """Hours the season window contains in each of its calendar months.

    Counted from the day-slots actually in the window, not from the calendar, so
    a window that starts mid-month gives that month its true partial length
    rather than a full one.

    NOTE this is the SHARED leap-year calendar, so February gets 29 days here
    for every season. Use :func:`season_month_window_hours` for the per-season
    denominator; this form is kept for the x-axis furniture, where one nominal
    month length is what is wanted.
    """
    months, mi_of_slot = season_month_axis(slots)
    out = np.zeros(len(months))
    for mi in mi_of_slot:
        out[mi] += 24.0
    return out


def resolve_phase_thresholds(args) -> dict:
    """Validated thresholds for whichever ``--phase-mode`` is selected.

    The returned dict always carries ``"mode"``; every consumer branches on it
    rather than sniffing which keys are present, so adding a third scheme later
    does not require finding all the places that guessed.

    ABSOLUTE mode raises via ``check_liquid_mixed_disjoint`` when liquid-only
    and mixed-phase would overlap. The whole condition is
    ``--liquid-iwp-max <= --mixed-iwp-min``: the categories are separated on the
    IWP axis alone, because LWP is unbounded above and no pair of LWP floors can
    pull them apart. ``--lwp-max-ice`` is validated separately, against the two
    LWP floors, so an ice-only scene can never also be a drawn one.

    FRACTION mode raises via ``check_fraction_thresholds_disjoint`` unless
    ``--liquid-fraction-min + --ice-fraction-min > 1``. Nothing else needs
    checking there: the three categories partition the water-bearing scenes by
    construction.
    """
    mode = getattr(args, "phase_mode", DEFAULT_PHASE_MODE)
    if mode not in PHASE_MODES:
        raise ValueError(f"unknown --phase-mode {mode!r}; "
                         f"choose from {list(PHASE_MODES)}")

    if mode == "fraction":
        kw = {
            "mode": "fraction",
            "liquid_fraction_min": float(args.liquid_fraction_min),
            "ice_fraction_min": float(args.ice_fraction_min),
            "min_lwp_g": float(args.min_lwp),
            "min_iwp_g": float(args.min_iwp),
        }
        check_fraction_thresholds_disjoint(kw["liquid_fraction_min"],
                                           kw["ice_fraction_min"])
        return kw

    kw = {
        "mode": "absolute",
        "liquid_lwp_min_g": float(args.liquid_lwp_min),
        "liquid_iwp_max_g": float(args.liquid_iwp_max),
        "mixed_lwp_min_g": float(args.mixed_lwp_min),
        "mixed_iwp_min_g": float(args.mixed_iwp_min),
        "ice_iwp_min_g": float(args.ice_iwp_min),
    }
    check_liquid_mixed_disjoint(kw["liquid_iwp_max_g"], kw["mixed_iwp_min_g"])

    lwp_max_ice = float(args.lwp_max_ice)
    lowest_drawn_floor = min(kw["liquid_lwp_min_g"], kw["mixed_lwp_min_g"])
    if lwp_max_ice > lowest_drawn_floor:
        raise ValueError(
            f"--lwp-max-ice {lwp_max_ice:g} exceeds the lowest liquid floor "
            f"{lowest_drawn_floor:g} g m-2, so an 'ice only' scene could also "
            f"be drawn as liquid only or mixed phase and would be counted "
            f"twice. Lower --lwp-max-ice below both LWP floors."
        )
    kw["lwp_max_ice_g"] = lwp_max_ice
    return kw


def phase_masks(lwp_g, iwp_g, phase_kw: dict) -> dict:
    """Every accumulated category, mutually exclusive and jointly exhaustive.

    Dispatches on ``phase_kw["mode"]``. The three named categories come from the
    shared module either way; ``none`` is built here as the remainder by
    CONSTRUCTION -- everything the other three did not claim -- so the four
    always partition the cloudy hours and the report's columns are guaranteed to
    add up whichever scheme is in use.

    In fraction mode ``none`` is exactly the scenes carrying no cloud water
    above the minimum paths, since the other three are already exhaustive over
    those that do.
    """
    finite = np.isfinite(lwp_g) & np.isfinite(iwp_g)

    if phase_kw["mode"] == "fraction":
        f = fraction_phase_masks(
            lwp_g, iwp_g,
            phase_kw["liquid_fraction_min"], phase_kw["ice_fraction_min"],
            phase_kw["min_lwp_g"], phase_kw["min_iwp_g"],
        )
        out = {"liquid": f["liquid"], "mixed": f["mixed"], "ice": f["ice"]}
    else:
        drawn = liquid_mixed_masks(
            lwp_g, iwp_g,
            phase_kw["liquid_lwp_min_g"], phase_kw["liquid_iwp_max_g"],
            phase_kw["mixed_lwp_min_g"], phase_kw["mixed_iwp_min_g"],
        )
        with np.errstate(invalid="ignore"):
            ice = (finite & (iwp_g > phase_kw["ice_iwp_min_g"])
                          & (lwp_g < phase_kw["lwp_max_ice_g"]))
        out = {"liquid": drawn["liquid"], "mixed": drawn["mixed"], "ice": ice}

    out["none"] = finite & ~out["liquid"] & ~out["mixed"] & ~out["ice"]
    return out


def lowest_drawn_lwp(phase_kw: dict) -> float:
    """Smallest LWP any drawn category can hold, for the log axis floor.

    Below it the log axis has nothing to show, by definition rather than by
    accident. In fraction mode that is the liquid minimum path: a scene whose
    liquid is under it contributes no LWP at all, so the lowest LWP that can
    appear on the figure is that floor.
    """
    if phase_kw["mode"] == "fraction":
        return float(phase_kw["min_lwp_g"])
    return float(min(phase_kw["liquid_lwp_min_g"], phase_kw["mixed_lwp_min_g"]))


def phase_definition_label(phase_kw: dict, mathtext: bool = True) -> str:
    """One-line statement of what the categories mean, for a figure subtitle."""
    u = "g m$^{-2}$" if mathtext else "g m-2"
    pk = phase_kw
    if pk["mode"] == "fraction":
        return (f"liquid only: LWP/CWP $\\geq$ {pk['liquid_fraction_min']:g}   |   "
                f"ice only: IWP/CWP $\\geq$ {pk['ice_fraction_min']:g}   |   "
                f"mixed: the rest   |   CWP = LWP + IWP above "
                f"{pk['min_lwp_g']:g}/{pk['min_iwp_g']:g} {u}"
                if mathtext else
                f"liquid only: LWP/CWP >= {pk['liquid_fraction_min']:g} | "
                f"ice only: IWP/CWP >= {pk['ice_fraction_min']:g} | "
                f"mixed: the rest | CWP = LWP + IWP above "
                f"{pk['min_lwp_g']:g}/{pk['min_iwp_g']:g} {u}")
    return (f"liquid only: LWP > {pk['liquid_lwp_min_g']:g}, IWP < "
            f"{pk['liquid_iwp_max_g']:g}   |   "
            f"mixed: LWP > {pk['mixed_lwp_min_g']:g}, IWP > "
            f"{pk['mixed_iwp_min_g']:g} {u}"
            if mathtext else
            f"liquid only: LWP > {pk['liquid_lwp_min_g']:g} and IWP < "
            f"{pk['liquid_iwp_max_g']:g} | mixed: LWP > "
            f"{pk['mixed_lwp_min_g']:g} and IWP > {pk['mixed_iwp_min_g']:g} {u}")


def ice_definition_label(phase_kw: dict, mathtext: bool = True) -> str:
    """How the ice-only category is defined, which the two modes state very differently."""
    u = "g m$^{-2}$" if mathtext else "g m-2"
    pk = phase_kw
    if pk["mode"] == "fraction":
        return (f"ice only: IWP/CWP $\\geq$ {pk['ice_fraction_min']:g}"
                if mathtext else f"ice only: IWP/CWP >= {pk['ice_fraction_min']:g}")
    return (f"ice only: IWP > {pk['ice_iwp_min_g']:g}, LWP < "
            f"{pk['lwp_max_ice_g']:g} {u}")


def phase_split_warning(col: dict) -> str | None:
    """Message naming a collapsed stack segment, or None when the split is real.

    Guards the one failure mode of the histogram that produces a
    plausible-looking picture rather than an error: thresholds under which one
    of the two STACKED categories is empty in practice. The bars then read as a
    single colour and the legend reports a handful of hours, with nothing to say
    that the thresholds and not the atmosphere are responsible.

    The classic absolute-mode case is an ice ceiling near ERA5's quantisation
    scale, where "liquid only" comes to mean "not a single quantum of ice" -- a
    condition almost no overcast Arctic column meets. See
    ERA5_TYPICAL_PATH_QUANTUM_G. In fraction mode the equivalent is a liquid
    share cut so close to 1 that any trace of ice disqualifies a cloud, which is
    why --min-iwp matters as much as --liquid-fraction-min there.

    Summed over the five surface classes only. The ARM site cell sits inside one
    of them, so including it would count its hours twice.
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
        msg = (f"the '{name}' segment holds {100 * share:.3f}% of the drawn "
               f"hours, so the stack is effectively all {other}")
        if pk["mode"] == "fraction":
            if name == "liquid only":
                msg += (f". --liquid-fraction-min "
                        f"{pk['liquid_fraction_min']:g} may be strict enough, "
                        f"or --min-iwp {pk['min_iwp_g']:g} g m-2 permissive "
                        f"enough, that a trace of ice disqualifies almost every "
                        f"cloud. Raising --min-iwp is usually the fix")
            else:
                msg += (f". --liquid-fraction-min "
                        f"{pk['liquid_fraction_min']:g} and --ice-fraction-min "
                        f"{pk['ice_fraction_min']:g} may between them leave "
                        f"almost nothing in the middle")
        elif (name == "liquid only"
                and pk["liquid_iwp_max_g"] <= 3 * ERA5_TYPICAL_PATH_QUANTUM_G):
            msg += (f". --liquid-iwp-max {pk['liquid_iwp_max_g']:g} g m-2 is "
                    f"within three quanta of ERA5's typical "
                    f"{ERA5_TYPICAL_PATH_QUANTUM_G:.4f} g m-2 quantisation "
                    f"step, so 'liquid only' is asking for literally zero ice. "
                    f"Try --liquid-iwp-max {DEFAULT_LIQUID_IWP_MAX:g}")
        elif name == "liquid only":
            msg += (f". --liquid-lwp-min {pk['liquid_lwp_min_g']:g} g m-2 may "
                    f"be high enough, or --liquid-iwp-max "
                    f"{pk['liquid_iwp_max_g']:g} strict enough, that almost no "
                    f"cloud qualifies")
        else:
            msg += (f". --mixed-iwp-min {pk['mixed_iwp_min_g']:g} or "
                    f"--mixed-lwp-min {pk['mixed_lwp_min_g']:g} g m-2 may be "
                    f"high enough that almost no cloud reaches them")
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
    axis running from a 5 g m-2 LWP floor the edges are 5, 7.775, 12.09, ...,
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
    That keeps the axis readable for any LWP floor / --lwp-lin-max the caller
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
def season_month_axis(slots: list[tuple[int, int]]):
    """Calendar months spanned by the season, and each day-slot's index into it.

    Returns ``(months, mi_of_slot)`` where ``months`` is the ordered list of
    calendar months the window touches -- [8, 9, 10, 11, 12, 1, 2, 3] for a
    default Aug-Mar season, in SEASON order rather than calendar order, so
    January follows December -- and ``mi_of_slot[d]`` is the position in that
    list of day-of-season ``d``.

    Ordered by first appearance rather than sorted, because a wrapping window
    would otherwise put January first and draw the season backwards.
    """
    slot_month = [m for m, _ in slots]
    months = list(dict.fromkeys(slot_month))
    index = {m: i for i, m in enumerate(months)}
    return months, np.array([index[m] for m in slot_month], dtype=np.intp)


def build_histograms(ds, lsm: np.ndarray, args, layout: dict,
                     wanted_idx: list[int], edge_sets: dict,
                     phase_kw: dict, sweep_values: np.ndarray) -> dict:
    """Accumulate the per-season LWP histograms in one streaming pass.

    Both bin scales and every phase are filled from the same blocks, so the
    archive is read once no matter how many copies of the figure are wanted.
    Only the seasons in ``wanted_idx`` are read at all.

    Returns a dict of arrays indexed ``[season, class, phase, bar]`` plus the
    denominators and diagnostics the report and the figures need.
    """
    slots = layout["slots"]
    # The histogram itself pools the whole window, but the monthly bar chart
    # needs each step's calendar month, so the day-of-season axis is carried.
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    uniq_seasons = layout["seasons"]
    months, mi_of_slot = season_month_axis(slots)

    wanted = np.zeros(len(uniq_seasons), dtype=bool)
    wanted[wanted_idx] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]
    # Restricting the UTC hours narrows the SAMPLE, not the window: every
    # denominator below counts only the kept steps, so the per-season and
    # per-month hours still read as "hours per season at this sampling rate"
    # and stay comparable with an unrestricted run.
    use_step = use_step & utc_hour_mask(ds, args)
    if not use_step.any():
        raise ValueError("--utc-hours removed every time step in the window")

    # The ARM site rides along as one extra slot on the class axis. It is NOT a
    # sixth class -- it is already inside whichever class it falls in -- so it
    # never enters an area share that is meant to sum to 100%.
    site_mask, site_lat, site_lon = site_cell_mask(ds)
    site_code = len(CLASS_ORDER)
    # And one more slot for EVERY valid cell, which the monthly bar chart uses
    # as its default. Accumulated as its own selector rather than summed from
    # the five classes afterwards, so it stays correct even when some cell-hours
    # are unclassified and the five do not in fact cover the domain.
    all_code = len(CLASS_ORDER) + 1
    n_class = len(CLASS_ORDER) + 2
    n_phase = len(PHASE_ORDER_ACC)
    n_season = len(uniq_seasons)
    n_month = len(months)

    hist = {
        scale: np.zeros((n_season, n_class, n_phase, n_bars(edges)))
        for scale, edges in edge_sets.items()
    }
    # One quantile accumulator PER DRAWN PHASE, so liquid-only and mixed-phase
    # each get their own median rather than sharing one over the pooled bars.
    # The two distributions are quite different -- a liquid-only deck is not a
    # mixed-phase one with the ice removed -- so a single pooled median
    # describes neither.
    n_stack = len(PHASE_STACK)
    qhist = np.zeros((n_season, n_class, n_stack, len(QUANTILE_EDGES_G) + 1))

    # Denominators and context, all area-weighted cell-hours.
    w_class = np.zeros((n_season, n_class))       # class present at all
    w_domain = np.zeros(n_season)                 # every cell, every step
    w_cloudy = np.zeros((n_season, n_class))      # class and cloudy
    w_steps = np.zeros(n_season)                  # time steps read per season

    # Month-resolved, for the monthly phase bar chart. The denominator is the
    # class's own valid cell-hours in that month, so a fraction is an occupancy:
    # "of the hours a cell of this class existed in October, what share had a
    # cloud of this phase overhead".
    w_phase_month = np.zeros((n_season, n_month, n_class, n_phase))
    w_valid_month = np.zeros((n_season, n_month, n_class))

    # --min-lwp sweep. Filled only in fraction mode -- see the note at
    # DEFAULT_SWEEP_LWP_MIN for why absolute mode has no single knob to sweep.
    sweep_lwp = (sweep_values if phase_kw["mode"] == "fraction"
                 else np.empty(0))
    n_sweep = sweep_lwp.size
    n_sph = len(SWEEP_PHASES)
    w_sweep = np.zeros((n_sweep, n_season, n_month, N_SWEEP_CLASS, n_sph))
    w_sweep_site = np.zeros((n_sweep, n_season, n_month, n_sph))
    sweep_phase_index = {p: i for i, p in enumerate(SWEEP_PHASES)}

    # cos(latitude) by default. The weighting decides what "a typical cell of
    # this class" means: 'area' answers "per unit AREA of the class", 'uniform'
    # answers "per GRID CELL of the class". They differ here because the classes
    # are latitude-structured -- pack ice sits north, where a 0.25 deg cell is
    # half the area of one at the southern edge -- so uniform weighting
    # over-represents the northern part of an ice class relative to its area.
    if getattr(args, "cell_weighting", "area") == "uniform":
        weights_2d = np.ones((ds.sizes["latitude"], ds.sizes["longitude"]))
    else:
        weights_2d = area_weights_2d(ds["latitude"].values,
                                     ds.sizes["longitude"])
    w_per_step = float(weights_2d.sum())

    n_unclassified = 0
    n_precip_removed = 0.0     # cloudy cell-hours the filter dropped
    n_cloudy_before = 0.0      # cloudy cell-hours before it
    site_class_counts = np.zeros(len(CLASS_ORDER) + 1, dtype=np.int64)

    read_vars = list(REQUIRED_VARS)
    liquid_var = getattr(args, "liquid_var", DEFAULT_LIQUID_VAR)
    if liquid_var not in read_vars:
        read_vars.append(liquid_var)
    if args.no_precip:
        read_vars += [v for v in PRECIP_SOURCE_VARS[args.precip_var]
                      if v not in read_vars]
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        sl = slice(i0, i0 + n_t)
        keep = use_step[sl]
        if not keep.any():
            continue
        si = s_idx[sl][keep]
        mi = mi_of_slot[dos[sl][keep]]                      # month of each step
        flat_sm = si * n_month + mi                         # (season, month)

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

        # The "liquid" axis is whichever field --liquid-var names; everything
        # downstream -- phase masks, LWP bins, the sweep -- reads this one array,
        # so tcslw flows through the identical logic rather than a parallel path.
        tclw_g = block[liquid_var].values[keep] * 1000.0    # kg m-2 -> g m-2
        tciw_g = block["tciw"].values[keep] * 1000.0
        tcc = block["tcc"].values[keep]

        valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw_g)
        # A precipitating scene is removed from the CLOUDY population, not
        # reclassified: it stops counting toward every phase and toward the
        # cloudy total alike, so the four phases still partition what is left.
        raining = precip_mask(block, keep, args)
        n_precip_removed += float(np.count_nonzero(raining & valid
                                                   & (tcc >= args.min_cloud_fraction)))
        n_cloudy_before += float(np.count_nonzero(valid
                                                  & (tcc >= args.min_cloud_fraction)))
        cloudy = valid & (tcc >= args.min_cloud_fraction) & ~raining
        phases = phase_masks(tclw_g, tciw_g, phase_kw)

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

        # --- the --min-lwp sweep ------------------------------------------
        # One np.bincount per threshold fills all six class slots at once,
        # because `classes` is already a partition of every cell. Looping class
        # by class would cost six times as much for the same numbers.
        if n_sweep:
            # UNCLASSIFIED (-1) folded onto its own slot so the six are a true
            # partition and "all cells" can be their exact sum.
            cls_idx = np.where(classes < 0, SWEEP_UNCLASSIFIED_SLOT,
                               classes).astype(np.intp)
            base = ((si[:, None, None] * n_month + mi[:, None, None])
                    * N_SWEEP_CLASS + cls_idx) * n_sph
            w_cloudy_cell = np.where(cloudy, w, 0.0)
            site_base = (si * n_month + mi) * n_sph
            w_site_cell = w_cloudy_cell[:, site_mask][:, 0]
            for ti, thr in enumerate(sweep_lwp):
                f = fraction_phase_masks(
                    tclw_g, tciw_g,
                    phase_kw["liquid_fraction_min"], phase_kw["ice_fraction_min"],
                    float(thr), phase_kw["min_iwp_g"],
                )
                label = np.full(classes.shape, sweep_phase_index["none"],
                                dtype=np.intp)
                label[f["liquid"]] = sweep_phase_index["liquid"]
                label[f["ice"]] = sweep_phase_index["ice"]
                label[f["mixed"]] = sweep_phase_index["mixed"]
                w_sweep[ti] += np.bincount(
                    (base + label).ravel(), weights=w_cloudy_cell.ravel(),
                    minlength=n_season * n_month * N_SWEEP_CLASS * n_sph,
                ).reshape(n_season, n_month, N_SWEEP_CLASS, n_sph)
                w_sweep_site[ti] += np.bincount(
                    site_base + label[:, site_mask][:, 0],
                    weights=w_site_cell,
                    minlength=n_season * n_month * n_sph,
                ).reshape(n_season, n_month, n_sph)

        selectors = [(CLASS_CODES[name], classes == CLASS_CODES[name])
                     for name in CLASS_ORDER]
        selectors.append((site_code, np.broadcast_to(site_mask, classes.shape)))
        selectors.append((all_code, np.ones(classes.shape, dtype=bool)))

        for code, in_class in selectors:
            wc = np.where(in_class & valid, w, 0.0)
            np.add.at(w_class, (si, code), wc.sum(axis=(1, 2)))
            np.add.at(w_cloudy, (si, code), (wc * cloudy).sum(axis=(1, 2)))

            # Monthly occupancy. Reduced over cells first, so the per-step
            # totals are a short (n_t,) vector and the (season, month) grouping
            # is one small bincount rather than a pass over the full grid.
            w_valid_month[:, :, code] += np.bincount(
                flat_sm, weights=wc.sum(axis=(1, 2)),
                minlength=n_season * n_month).reshape(n_season, n_month)
            for pi, phase in enumerate(PHASE_ORDER_ACC):
                w_phase_month[:, :, code, pi] += np.bincount(
                    flat_sm, weights=(wc * cloudy * phases[phase]).sum(axis=(1, 2)),
                    minlength=n_season * n_month).reshape(n_season, n_month)

            in_cloud = in_class & cloudy
            if not in_cloud.any():
                continue
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
                    qi = PHASE_STACK.index(phase)
                    flat = s_sel * n_qbar + qbar[sel]
                    qhist[:, code, qi] += np.bincount(
                        flat, weights=w_sel, minlength=n_season * n_qbar,
                    ).reshape(n_season, n_qbar)

    return {
        "precip_removed": n_precip_removed,
        "cloudy_before_precip": n_cloudy_before,
        "hist": hist,
        "qhist": qhist,
        "w_class": w_class,
        "w_cloudy": w_cloudy,
        "w_domain": w_domain,
        "w_steps": w_steps,
        "w_phase_month": w_phase_month,
        "w_valid_month": w_valid_month,
        "w_sweep": w_sweep,
        "w_sweep_site": w_sweep_site,
        "sweep_lwp": sweep_lwp,
        "sweep_spacing": getattr(args, "sweep_spacing", DEFAULT_SWEEP_SPACING),
        "months": months,
        "all_code": all_code,
        "slots": slots,
        "seasons": uniq_seasons,
        "site_code": site_code,
        "site_lat": site_lat,
        "site_lon": site_lon,
        "site_class_counts": site_class_counts,
        "n_unclassified": n_unclassified,
    }


def to_hours_per_season(sec: dict, keep_idx: list[int], edge_sets: dict,
                        layout: dict | None = None) -> dict:
    """Convert accumulated weights to hours per season and average the seasons.

    See the module docstring for the normalisation. In one line: divide by the
    class's own cell-hour total so the answer is per cell rather than per class,
    then multiply by the nominal length of the season window so a partially
    sampled season is scaled up rather than counted short.
    """
    n_slot = len(sec["slots"])
    # Per season, not one number for all of them: a common-year Oct-Mar window
    # is 182 days and a leap-year one 183. With the season's own length here, a
    # COMPLETE season has counts/valid == 1 and passes through unscaled, which
    # is the property that makes the scaling correct rather than merely small.
    if layout is not None and "days_per_season" in layout:
        season_hours = season_window_hours(layout, keep_idx)      # (season,)
    else:
        season_hours = np.full(len(keep_idx), n_slot * 24.0 * HOURS_PER_STEP)

    w_class = sec["w_class"][keep_idx]                     # (season, class)
    denom = np.where(w_class > 0, w_class, np.nan)

    hours = {}
    for scale in edge_sets:
        h = sec["hist"][scale][keep_idx]                   # (s, class, phase, bar)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_season = (h / denom[:, :, None, None]
                          * season_hours[:, None, None, None])
        hours[scale] = {
            "per_season": per_season,                      # kept for the spread
            "mean": nanmean_quiet(per_season, axis=0),     # (class, phase, bar)
        }

    with np.errstate(invalid="ignore", divide="ignore"):
        cloudy_hours = (sec["w_cloudy"][keep_idx] / denom
                        * season_hours[:, None])
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
    q_pooled = sec["qhist"][keep_idx].sum(axis=0)          # (class, phase, qbar)
    median_lwp_g = np.array([
        [weighted_median_from_bins(q_pooled[c, i], QUANTILE_EDGES_G)
         for i in range(q_pooled.shape[1])]
        for c in range(q_pooled.shape[0])
    ])                                                     # (class, phase)

    # Monthly phase occupancy: the share of a class's valid cell-hours in each
    # calendar month that held a cloud of each phase. Already a fraction, so no
    # season-length scaling applies and a partly-sampled month is represented by
    # the hours it does have rather than being scaled up.
    wpm = sec["w_phase_month"][keep_idx]              # (s, month, class, phase)
    wvm = sec["w_valid_month"][keep_idx]              # (s, month, class)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_per_season = np.where(wvm[..., None] > 0,
                                   wpm / np.where(wvm[..., None] > 0,
                                                  wvm[..., None], 1.0),
                                   np.nan)
    month_fraction = {
        "per_season": frac_per_season,                     # kept for the spread
        "mean": nanmean_quiet(frac_per_season, axis=0),    # (month, class, phase)
    }

    # --- the --min-lwp sweep --------------------------------------------
    # Class axis: the five classes and UNCLASSIFIED, plus two derived slots
    # appended here -- the ARM site cell, and "all cells" as the exact sum over
    # the six. Indexed the same way as everything else downstream via
    # sweep_class_slot().
    ws = sec["w_sweep"][:, keep_idx]                  # (thr, s, month, cls, ph)
    sweep = None
    if ws.size:
        wss = sec["w_sweep_site"][:, keep_idx]        # (thr, s, month, ph)
        all_cls = ws.sum(axis=3, keepdims=True)       # exact: the six partition
        ws_full = np.concatenate([ws, wss[:, :, :, None, :], all_cls], axis=3)

        # Denominators do NOT depend on the threshold: valid cell-hours are a
        # property of the grid and the classification, not of --min-lwp.
        den_cls = sec["w_valid_month"][keep_idx]      # (s, month, class7)
        order = [CLASS_CODES[n] for n in CLASS_ORDER]
        den = np.concatenate([
            den_cls[:, :, order],                                    # 5 classes
            den_cls[:, :, order].sum(axis=2, keepdims=True) * 0.0,   # unclass.
            den_cls[:, :, [sec["site_code"], sec["all_code"]]],      # site, all
        ], axis=2)
        # The unclassified slot has no denominator of its own in
        # w_valid_month; give it the residual so its fraction is still defined.
        den[:, :, SWEEP_UNCLASSIFIED_SLOT] = np.maximum(
            den_cls[:, :, sec["all_code"]] - den_cls[:, :, order].sum(axis=2),
            0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den[None, ..., None] > 0,
                            ws_full / np.where(den[None, ..., None] > 0,
                                               den[None, ..., None], 1.0),
                            np.nan)                   # (thr, s, month, cls, ph)
        month_h = month_window_hours(sec["slots"])    # (month,)
        sweep = {
            "lwp": sec["sweep_lwp"],
            "spacing": sec["sweep_spacing"],
            "month_fraction": nanmean_quiet(frac, axis=1),   # (thr, month, cls, ph)
            "month_hours_axis": month_h,
            # Seasonal totals pool the months before dividing, so a long month
            # counts for more than a short one -- which is what "fraction of the
            # season" means. Averaging the monthly fractions instead would
            # silently give February the same weight as December.
            "season_fraction": nanmean_quiet(
                np.where(den.sum(axis=1)[None, ..., None] > 0,
                         ws_full.sum(axis=2)
                         / np.where(den.sum(axis=1)[None, ..., None] > 0,
                                    den.sum(axis=1)[None, ..., None], 1.0),
                         np.nan),
                axis=1),                                      # (thr, cls, ph)
        }

    return {
        "sweep": sweep,
        "hours": hours,
        "n_seasons": len(keep_idx),
        "season_hours": season_hours,                      # (season,) hours
        "month_hours": (season_month_window_hours(layout, keep_idx)
                        if layout is not None
                        and "month_days_per_season" in layout
                        else None),                        # (season, month)
        "month_fraction": month_fraction,
        "months": sec["months"],
        "all_code": sec["all_code"],
        "cloudy_hours": nanmean_quiet(cloudy_hours, axis=0),
        "cloudy_hours_per_season": cloudy_hours,
        "area_pct": nanmean_quiet(area_pct, axis=0),
        "median_lwp_g": median_lwp_g,
        "site_code": sec["site_code"],
        "precip_removed": sec.get("precip_removed", 0.0),
        "cloudy_before_precip": sec.get("cloudy_before_precip", 0.0),
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


# Width of the notes column, as a fraction of one panel's width. Wide enough for
# a threshold line to sit on one row without wrapping, narrow enough that it does
# not steal the figure from the data.
NOTE_COL_WIDTH = 0.62

# Style of the side note box, shared by every figure here so the pages match.
NOTE_FONTSIZE = 8.8
NOTE_LINESPACING = 1.6
NOTE_BOX_PAD = 0.6                      # boxstyle pad, in units of the font size
NOTE_BOX = dict(boxstyle=f"round,pad={NOTE_BOX_PAD}", facecolor="#f5f5f2",
                edgecolor="#bfbfbf", linewidth=0.8)

# Breathing room left under the note when the figure is grown to fit it, and the
# cap on how many times that growth is retried. Four is far more than the two
# passes it takes to converge in practice; it exists so a pathological note can
# never spin.
NOTE_FIT_PAD_IN = 0.08
NOTE_FIT_TOL_IN = 0.02
NOTE_FIT_MAX_ITER = 6

# Fallback only, for a backend that will not hand over a renderer: the note
# height from font metrics alone. MEASURED against the rendered box, this
# over-estimates by 5-9% over 5-30 lines, which is the right direction for a
# fallback -- it grows the figure slightly too much rather than too little.
NOTE_HEADROOM_IN = 0.65


# Median lines: one per drawn phase, told apart by dash pattern rather than by
# colour, because a line in the bar's own colour disappears against the bar.
MEDIAN_LINE_COLOR = "#B2182B"
MEDIAN_LINE_STYLE: dict[str, str] = {"liquid": ":", "mixed": "--"}


def wrap_note(text: str, width: int = 34) -> str:
    """Hard-wrap a sentence to the notes column."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def phase_note_lines(pk: dict, args, col: dict) -> list[str]:
    """The season, cloud, and phase definitions, as lines for the side note."""
    lines = [
        f"Season {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
        f"{args.season_end[0]:02d}-{args.season_end[1]:02d}",
        f"  ({window_label(col['season_hours'])} per season)",
        "",
        f"Cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}",
        "",
    ]
    if pk["mode"] == "fraction":
        lines += [
            "CWP = LWP + IWP, counting",
            f"only paths above {pk['min_lwp_g']:g} / "
            f"{pk['min_iwp_g']:g} g m$^{{-2}}$",
            "",
            f"Liquid only: LWP/CWP $\\geq$ {pk['liquid_fraction_min']:g}",
            f"Ice only: IWP/CWP $\\geq$ {pk['ice_fraction_min']:g}",
            "Mixed phase: everything else",
        ]
    else:
        lines += [
            f"Liquid only: LWP > {pk['liquid_lwp_min_g']:g},",
            f"  IWP < {pk['liquid_iwp_max_g']:g} g m$^{{-2}}$",
            f"Mixed phase: LWP > {pk['mixed_lwp_min_g']:g},",
            f"  IWP > {pk['mixed_iwp_min_g']:g} g m$^{{-2}}$",
            f"Ice only: IWP > {pk['ice_iwp_min_g']:g},",
            f"  LWP < {pk['lwp_max_ice_g']:g} g m$^{{-2}}$",
        ]
    return lines


def panel_grid_with_notes(n_r: int, n_c: int, panel_w: float, panel_h: float,
                          sharex: bool = True, sharey: bool = True):
    """A panel grid plus a dedicated notes column on the right.

    Returns ``(fig, axes, ax_note)``. ``ax_note`` is an invisible axes spanning
    every row of an extra narrow column; write into it with axes coordinates.

    Built from an explicit gridspec rather than by dropping a ``fig.text`` at
    x > 1, because a figure-level text outside the axes is invisible to
    constrained_layout: it would be clipped in the notebook's inline display and
    only reappear on save, where ``bbox_inches="tight"`` expands the canvas. A
    reserved column is laid out like anything else and looks the same in both.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(panel_w * (n_c + NOTE_COL_WIDTH), panel_h * n_r),
                     constrained_layout=True)
    gs = fig.add_gridspec(n_r, n_c + 1,
                          width_ratios=[1.0] * n_c + [NOTE_COL_WIDTH])
    axes = []
    for r in range(n_r):
        for c in range(n_c):
            kw = {}
            if axes:
                if sharex:
                    kw["sharex"] = axes[0]
                if sharey:
                    kw["sharey"] = axes[0]
            axes.append(fig.add_subplot(gs[r, c], **kw))
    ax_note = fig.add_subplot(gs[:, -1])
    ax_note.axis("off")
    return fig, np.array(axes), ax_note


def suptitle_over_panels(fig, text: str, n_c: int, **kw) -> None:
    """Centre the suptitle over the panel area, ignoring the notes column.

    ``fig.suptitle`` centres on the whole canvas, which with a reserved notes
    column pushes the heading right and lets it collide with the note box. The
    panels occupy the first ``n_c`` of ``n_c + NOTE_COL_WIDTH`` width units.
    """
    fig.suptitle(text, x=0.5 * n_c / (n_c + NOTE_COL_WIDTH), **kw)


def note_height_in(text: str) -> float:
    """Height the rendered note needs, in inches, from the font metrics alone.

    Computed rather than measured so it is available BEFORE a draw, which is
    what lets :func:`draw_notes` resize the figure without rendering it twice.
    Line count times font size times line spacing, plus the box padding at top
    and bottom (boxstyle pad is in units of the font size).
    """
    n_lines = text.count("\n") + 1
    return ((n_lines * NOTE_FONTSIZE * NOTE_LINESPACING
             + 2 * NOTE_BOX_PAD * NOTE_FONTSIZE) / 72.0)


def fit_note_in_figure(ax_note, t) -> None:
    """Grow the figure until the note fits inside its own axes.

    Measured rather than predicted. What has to fit is the note box inside
    ``ax_note``, and the space available to it is the row height LESS whatever
    constrained_layout gave the suptitle and the outer padding -- neither of
    which is known before a draw. So: draw, measure the shortfall, grow by
    exactly that, repeat. The note's height in inches is fixed while the axes
    only grows, so this converges; two passes is the usual cost.

    CALL THE SUPTITLE FIRST. The measurement is only as good as the layout it
    reads, and a suptitle added afterwards takes its height back out of the same
    rows -- leaving the note clipped by precisely that much.
    """
    fig = ax_note.get_figure()
    h0 = fig.get_size_inches()[1]        # never shrink below what the caller asked for
    if fig._suptitle is None:
        # The suptitle takes its space out of the same rows the note sits in,
        # so measuring before it exists would under-report the shortfall and
        # leave the note clipped by exactly the title's height. Every figure in
        # this module calls suptitle_over_panels() first; this catches a new one
        # that does not, instead of silently mis-fitting.
        warnings.warn("draw_notes() ran before the suptitle was set; the note "
                      "may be clipped. Call suptitle_over_panels() first.",
                      RuntimeWarning, stacklevel=3)
    for _ in range(NOTE_FIT_MAX_ITER):
        try:
            with warnings.catch_warnings():
                # A figure that is still too small can make constrained_layout
                # give up and say so. That is the state this loop exists to
                # correct, and it is gone by the next iteration, so the warning
                # is noise here -- it would fire on the intermediate draw and
                # not on the figure the caller actually gets.
                warnings.filterwarnings(
                    "ignore", message=".*constrained_layout not applied.*",
                    category=UserWarning)
                fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except AttributeError:
            # No renderer to ask. Fall back to the font-metric estimate, which
            # errs high, and take it in one step.
            w_in, h_in = fig.get_size_inches()
            needed = note_height_in(t.get_text()) + NOTE_HEADROOM_IN
            if needed > h_in:
                fig.set_size_inches(w_in, needed, forward=False)
            return
        box = t.get_bbox_patch() or t
        # Positive: the note hangs below the axes by this much. Negative: slack.
        short_in = ((ax_note.get_window_extent(renderer).y0
                     - box.get_window_extent(renderer).y0) / fig.dpi)
        w_in, h_in = fig.get_size_inches()
        adjust = short_in + NOTE_FIT_PAD_IN
        # Shrinking back matters because the FIRST measurement is taken while
        # the note is still overflowing -- and therefore while constrained
        # layout is squeezing the very axes being measured. The shortfall reads
        # too large, the figure overshoots, and without this the figure ends up
        # 0.3-0.7 in taller than the note needs. Never below the caller's own
        # height, so a note that always fitted cannot make the figure smaller.
        if abs(adjust) <= NOTE_FIT_TOL_IN:
            return
        new_h = max(h0, h_in + adjust)
        if abs(new_h - h_in) <= NOTE_FIT_TOL_IN:
            return
        fig.set_size_inches(w_in, new_h, forward=False)

    # Ran out of iterations. Whatever the last adjustment was, the figure must
    # not be left clipping, so make one unconditional growth pass.
    _grow_note_clear(ax_note, t)


def _grow_note_clear(ax_note, t) -> None:
    """Last-resort growth: guarantee the note is not clipped, at any height."""
    fig = ax_note.get_figure()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*constrained_layout not applied.*",
                category=UserWarning)
            fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        return
    box = t.get_bbox_patch() or t
    short_in = ((ax_note.get_window_extent(renderer).y0
                 - box.get_window_extent(renderer).y0) / fig.dpi)
    if short_in > 0:
        w_in, h_in = fig.get_size_inches()
        fig.set_size_inches(w_in, h_in + short_in + NOTE_FIT_PAD_IN,
                            forward=False)


def draw_notes(ax_note, lines, title: str | None = None):
    """Render the side note, growing the figure if the note will not fit.

    ``None`` entries are dropped, so a caller can build the list with optional
    rows inline. An empty STRING is kept and becomes a blank line -- that is how
    the note is grouped into blocks, so the two must not be conflated.

    THE RESIZE IS NOT COSMETIC, and it is the fix for a real failure. The note
    is one text artist anchored to the TOP of ``ax_note``, so a note taller than
    that axes has a tight bbox hanging below the bottom of its grid row.
    constrained_layout honours tight bboxes, so it responds by shrinking the
    row -- and the row holds the DATA PANEL too. The panel collapses toward the
    top of the canvas while the note sits apparently fine beside it, which reads
    as "the figure did not render" rather than as "the note is two lines too
    long".

    MEASURED, one 7.6 x 5.2 in panel plus the notes column, panel height as a
    fraction of the row:

        note lines      22      26      30
        plain         0.85    0.71    0.43
        with twiny    0.79    0.56    0.27

    A twin axis roughly doubles the loss, because it adds its own margin demand
    to the same row. That combination -- 26 lines AND a twiny -- is what left
    the duration figure at 0.33 of its row, drawn as a sliver at the top of an
    otherwise empty canvas.

    ``bbox_inches="tight"`` hides all of this on save, because it crops the
    canvas back to the artists; the collapse is visible only in the figure's
    own geometry, which is what a notebook displays inline.
    """
    body = "\n".join(ln for ln in lines if ln is not None)
    text = f"{title}\n{body}" if title else body
    t = ax_note.text(0.0, 1.0, text, transform=ax_note.transAxes,
                     va="top", ha="left", fontsize=NOTE_FONTSIZE,
                     linespacing=NOTE_LINESPACING, bbox=NOTE_BOX)
    fit_note_in_figure(ax_note, t)
    return t


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
    fig, axes, ax_note = panel_grid_with_notes(n_r, n_c, 4.6, 3.9)

    ticks, labels = bar_ticks(edges_g, log_spaced, has_underflow)
    phase_idx = {p: PHASE_ORDER_ACC.index(p) for p in PHASE_STACK}
    pk = col["phase_kw"]

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
        pct_season = 100.0 * total_h / float(np.mean(col["season_hours"]))
        # No area share in the title. Three of the five classes are defined by
        # sea ice concentration and migrate through the season, so any single
        # number is a time average of a moving quantity -- which invites being
        # read as a fixed property of the panel. The report still carries it,
        # labelled as a mean.
        where = "1 cell, inside another class" if is_site else None
        head = f"{label}   ({where})" if where else label
        # "liquid + mixed", never "liquid-bearing". With independent thresholds
        # the two categories no longer partition any simply-stated superset, so
        # the title names exactly what the bars hold and nothing more. The
        # remainder is the report's 'neither' column.
        # "of all hours", not "of the season": the denominator is every hour in
        # the window, CLEAR ONES INCLUDED, not the overcast subset. The algebra
        # makes this exact -- a bar is
        #     (phase cell-hours / valid cell-hours) x season_hours
        # so dividing the summed bars by season_hours cancels it, leaving
        # phase cell-hours / valid cell-hours. The conditional version, "of the
        # overcast hours, what share was liquid-bearing", is a different and
        # larger number, and lives in the report's 'liq+mix %' column.
        ax.set_title(f"{head}\n"
                     f"{total_h:,.0f} h liquid + mixed = "
                     f"{pct_season:.1f}% of all hours",
                     fontsize=10.5, pad=6,
                     fontweight="bold" if is_site else "normal",
                     color=SITE_COLOR if is_site else "black")

        # One median per drawn phase. A liquid-only deck and a mixed-phase one
        # have genuinely different LWP distributions, so a single median over
        # the pooled bars describes neither of them.
        handles, hlabels = ax.get_legend_handles_labels()
        for pi, phase in enumerate(PHASE_STACK):
            med = col["median_lwp_g"][code, pi]
            xm = value_to_bar_x(med, edges_g, log_spaced)
            if xm is None or xm < first_bar - 0.5:
                continue
            style = MEDIAN_LINE_STYLE[phase]
            ax.axvline(xm, color=MEDIAN_LINE_COLOR, lw=1.5, ls=style, zorder=6)
            hlabels.append(f"median LWP, {MONTH_BAR_LABELS[phase].lower()}"
                           f" = {med:,.3g} g m$^{{-2}}$")
            handles.append(plt.Line2D([], [], color=MEDIAN_LINE_COLOR, lw=1.5,
                                      ls=style))
        if share < args.min_class_area and not is_site:
            hlabels.append(f"class averages only {share:.2f}% of the domain")
            handles.append(plt.Line2D([], [], color="none"))
        ax.legend(handles, hlabels, fontsize=8.0, framealpha=0.85, loc="best")

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
    # The title says what the figure IS; every parameter that defines it goes in
    # the side note, so the heading never outruns the plot it labels.
    suptitle_over_panels(
        fig,
        f"Liquid-bearing cloud hours by LWP and surface class — {region}\n"
        f"{mode_label}"
        f"{', mean across seasons' if col['n_seasons'] > 1 else ''}",
        n_c, fontsize=12.5)
    draw_notes(
        ax_note,
        phase_note_lines(pk, args, col)
        + ["", f"Bins: {bin_note}", "",
           wrap_note("Panel % is of ALL hours in the window, clear ones "
                     "included. For the share of OVERCAST hours instead, see "
                     "the report's 'liq+mix %' column.", 34)]
        + ([] if degenerate is None else
           ["", "!! " + wrap_note(degenerate, 34)]))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {output_path}")
    return fig


# Bars drawn by the monthly figure, in order. Unlike the histogram, ice-only IS
# shown here: the x axis is a category, not LWP, so an ice cloud has somewhere to
# go and its seasonal rise is the point of the figure.
MONTH_BAR_PHASES: tuple[str, ...] = ("liquid", "ice", "mixed")

MONTH_BAR_LABELS: dict[str, str] = {
    "liquid": "Liquid only",
    "ice": "Ice only",
    "mixed": "Mixed phase",
}


def resolve_series_code(col: dict, surface_class: str | None) -> tuple[int, str]:
    """(class-axis index, label) for a named surface class, the site, or all.

    ``None`` or ``"all"`` selects every valid cell in the domain, which is the
    default for the monthly figure: the question it answers -- how the phase mix
    turns over through the season -- is about the region, and the histogram
    already carries the per-class breakdown.
    """
    if surface_class in (None, "all"):
        return col["all_code"], "all cells"
    if surface_class in (SITE_KEY, "site", "arm"):
        return col["site_code"], SITE_LABEL
    if surface_class in CLASS_CODES:
        return CLASS_CODES[surface_class], CLASS_LABELS[surface_class]
    raise KeyError(
        f"unknown surface class {surface_class!r}; choose from "
        f"{['all', SITE_KEY] + list(CLASS_ORDER)}"
    )


def fig_monthly_phase_fraction(A: Analysis, out_dir=None, dpi: int | None = None,
                               surface_class: str | None = None):
    """Monthly phase occupancy: one panel per month, three bars per panel.

    Each bar is the share of that month's cell-hours spent under a cloud of one
    phase -- liquid only, ice only, mixed phase -- averaged over the seasons,
    with the spread across seasons drawn as a whisker when there is more than
    one.

    The three bars do NOT sum to 100%. They are three of the four categories
    that partition the overcast hours (the fourth, "neither", is the remainder
    named in the report), and clear hours are in none of them. The panel
    subtitle states the overcast share so the gap is accounted for rather than
    left for the reader to wonder about.

    Panel grid is two rows by however many columns the season needs: 2 x 3 for
    an Oct-Mar window, 2 x 4 for Aug-Mar.

    ``surface_class`` picks which cells to average over -- ``None``/``"all"``
    for the whole domain, any name in ``CLASS_ORDER``, or ``"arm_site"``.
    """
    col, args = A.col, A.args
    code, series_label = resolve_series_code(col, surface_class)
    months = col["months"]
    mean_frac = col["month_fraction"]["mean"]          # (month, class, phase)
    per_season = col["month_fraction"]["per_season"]   # (s, month, class, phase)
    n_m = len(months)

    n_r = 2
    n_c = -(-n_m // n_r)              # ceil: 6 months -> 2x3, 8 -> 2x4
    fig, axes, ax_note = panel_grid_with_notes(n_r, n_c, 3.3, 3.7, sharex=False)

    x = np.arange(len(MONTH_BAR_PHASES))
    colors = [PHASE_COLORS[p] for p in MONTH_BAR_PHASES]
    labels = [MONTH_BAR_LABELS[p] for p in MONTH_BAR_PHASES]
    idx = [PHASE_ORDER_ACC.index(p) for p in MONTH_BAR_PHASES]

    ymax = 0.0
    for k, month in enumerate(months):
        ax = axes[k]
        y = 100.0 * np.array([mean_frac[k, code, i] for i in idx])
        y = np.nan_to_num(y)
        # Spread across seasons, drawn as a whisker rather than a symmetric
        # error bar: the quantity is a bounded fraction and its across-season
        # distribution is not symmetric near 0 or 100%.
        lo = hi = None
        if per_season.shape[0] > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                block = 100.0 * per_season[:, k, code, :][:, idx]   # (s, phase)
                lo = np.nanmin(block, axis=0)
                hi = np.nanmax(block, axis=0)
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0),
                                        np.maximum(hi - y, 0)],
                            fmt="none", ecolor="#333333", elinewidth=1.0,
                            capsize=4, zorder=5)
                ymax = max(ymax, float(np.nanmax(hi)))
        ymax = max(ymax, float(y.max()))

        ax.bar(x, y, width=0.62, color=colors, edgecolor="none", zorder=3)
        # Label above the WHISKER, not the bar: at the bar top it collides with
        # the upper cap wherever the across-season spread is wide, which is
        # exactly the months worth reading carefully.
        tops = y if hi is None else np.maximum(y, np.nan_to_num(hi, nan=0.0))
        for xi, v, top in zip(x, y, tops):
            ax.text(xi, top, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, zorder=6)

        overcast = 100.0 * float(np.nansum(
            [mean_frac[k, code, PHASE_ORDER_ACC.index(pp)]
             for pp in PHASE_ORDER_ACC]))
        ax.set_title(f"{calendar.month_name[month]}\n"
                     f"{overcast:.0f}% of hours overcast",
                     fontsize=10.5, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", labelsize=9.5)

    for ax in axes[n_m:]:
        ax.set_visible(False)
    for k in range(0, len(axes), n_c):
        if axes[k].get_visible():
            axes[k].set_ylabel("Share of ALL cell-hours\nin the month [%]",
                               fontsize=10.5)
    axes[0].set_ylim(0, max(5.0, ymax * 1.22))

    pk = col["phase_kw"]
    suptitle_over_panels(
        fig,
        f"Monthly cloud-phase occupancy — {args.region}, {series_label}\n"
        f"{A.mode_label}"
        f"{', mean across seasons' if col['n_seasons'] > 1 else ''}",
        n_c, fontsize=12.5)
    extra = ["", wrap_note("Bars are three of the four categories that "
                           "partition the overcast hours; the fourth is the "
                           "report's 'neither' column, and clear hours are in "
                           "none. They need not sum to the overcast share.", 34)]
    if col["n_seasons"] > 1:
        extra += ["", "Whiskers span the min-max", "across seasons."]
    draw_notes(ax_note, phase_note_lines(pk, args, col) + extra)

    if out_dir is not None:
        tag = "all" if surface_class in (None, "all") else str(surface_class)
        path = (Path(out_dir) /
                f"{args.region}_monthly_phase_fraction_{tag}_{A.tag}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig


# Sweep class axis: five classes, unclassified, then the two derived slots.
SWEEP_SITE_SLOT = N_SWEEP_CLASS
SWEEP_ALL_SLOT = N_SWEEP_CLASS + 1


def sweep_class_slot(surface_class: str | None) -> tuple[int, str]:
    """(sweep class-axis index, label) for a class name, the site, or all."""
    if surface_class in (None, "all"):
        return SWEEP_ALL_SLOT, "all cells"
    if surface_class in (SITE_KEY, "site", "arm"):
        return SWEEP_SITE_SLOT, SITE_LABEL
    if surface_class in CLASS_CODES:
        return CLASS_ORDER.index(surface_class), CLASS_LABELS[surface_class]
    raise KeyError(
        f"unknown surface class {surface_class!r}; choose from "
        f"{['all', SITE_KEY] + list(CLASS_ORDER)}")


def require_sweep(A: Analysis) -> dict:
    """The sweep data, or a message explaining why there is none."""
    sweep = A.col.get("sweep")
    if sweep is None:
        raise ValueError(
            "no --min-lwp sweep was accumulated. It exists only in "
            "phase_mode='fraction': absolute mode carries separate LWP floors "
            "for liquid-only and mixed-phase, so there is no single minimum to "
            "sweep. Re-run prepare(phase_mode='fraction').")
    return sweep


def _sweep_axes(ax, lwp, spacing: str = DEFAULT_SWEEP_SPACING,
                log_y: bool = False) -> None:
    """Axis furniture for a sweep panel.

    The x SCALE follows the sampling: points laid out linearly on a log axis (or
    the reverse) would bunch up and misrepresent where the resolution actually
    is.
    """
    if spacing == "log":
        ax.set_xscale("log")
    ax.set_xlim(lwp[0], lwp[-1])
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9.5)
    if log_y:
        ax.set_yscale("log")


def fig_sweep_monthly(A: Analysis, out_dir=None, dpi: int | None = None,
                      surface_class: str | None = None, as_hours: bool = False):
    """Phase occupancy against the minimum LWP threshold, one panel per month.

    Three lines per panel -- liquid only, ice only, mixed phase -- against
    ``--min-lwp`` on a log x axis. ``as_hours=False`` plots the share of the
    month's cell-hours (the same quantity the monthly bar chart shows);
    ``as_hours=True`` plots that share times the month's own length in the
    window, i.e. average hours per month.

    Every threshold on the axis was classified in the same pass over the
    archive, so the curves are exactly comparable -- no reload, no resampling,
    and identical cell-hours behind every point.
    """
    sweep = require_sweep(A)
    col, args = A.col, A.args
    slot, series_label = sweep_class_slot(surface_class)
    lwp = sweep["lwp"]
    months = col["months"]
    frac = sweep["month_fraction"]                    # (thr, month, cls, ph)
    month_h = sweep["month_hours_axis"]
    n_m = len(months)
    drawn = sweep_drawn_phases(args)

    n_r = 2
    n_c = -(-n_m // n_r)
    fig, axes, ax_note = panel_grid_with_notes(n_r, n_c, 3.5, 3.4, sharex=True,
                                               sharey=True)

    for k, month in enumerate(months):
        ax = axes[k]
        for phase in drawn:
            pi = SWEEP_PHASES.index(phase)
            y = frac[:, k, slot, pi] * (month_h[k] if as_hours else 100.0)
            ax.plot(lwp, y, color=PHASE_COLORS[phase], lw=1.9,
                    marker="o", markersize=2.8,
                    label=MONTH_BAR_LABELS[phase])
        ax.set_title(f"{calendar.month_name[month]}"
                     + (f"   ({month_h[k]:,.0f} h)" if as_hours else ""),
                     fontsize=10.5, pad=6)
        _sweep_axes(ax, lwp, sweep["spacing"])
        if k == 0:
            ax.legend(fontsize=8.5, framealpha=0.9, loc="best")

    for ax in axes[n_m:]:
        ax.set_visible(False)
    ylab = ("Hours per month\nper grid cell" if as_hours else
            "Share of ALL cell-hours\nin the month [%]")
    for k in range(0, len(axes), n_c):
        if axes[k].get_visible():
            axes[k].set_ylabel(ylab, fontsize=10.5)
    for k in range(len(axes) - n_c, len(axes)):
        if axes[k].get_visible():
            axes[k].set_xlabel("minimum LWP [g m$^{-2}$]", fontsize=10.5)

    what = "hours per month" if as_hours else "share of the month"
    suptitle_over_panels(
        fig,
        f"Cloud phase vs the minimum LWP threshold — {args.region}, "
        f"{series_label}\n{A.mode_label}"
        f"{', mean across seasons' if col['n_seasons'] > 1 else ''}"
        f"   |   {what}",
        n_c, fontsize=12.5)
    draw_notes(ax_note, sweep_note_lines(
        col, args,
        "y axis: hours per month = the month's share of cell-hours times its "
        "own length in the window." if as_hours else
        "Denominator: ALL cell-hours in that month, clear ones included."))

    if out_dir is not None:
        tag = "all" if surface_class in (None, "all") else str(surface_class)
        kind = "hours" if as_hours else "fraction"
        path = (Path(out_dir) / f"{args.region}_sweep_monthly_{kind}_{tag}_"
                                f"{A.tag}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig


def fig_sweep_season(A: Analysis, out_dir=None, dpi: int | None = None,
                     surface_class: str | None = None, as_hours: bool = False):
    """The same sweep pooled over the whole season, in one panel.

    ``as_hours=False`` plots the share of all cell-hours in the season window;
    ``as_hours=True`` multiplies by the window length to give hours per season.

    Months are pooled BEFORE dividing, so a 31-day month carries more weight
    than a 28-day one -- which is what "fraction of the season" means. Averaging
    the six monthly curves instead would quietly give February December's
    weight.
    """
    sweep = require_sweep(A)
    col, args = A.col, A.args
    slot, series_label = sweep_class_slot(surface_class)
    lwp = sweep["lwp"]
    frac = sweep["season_fraction"]                   # (thr, cls, ph)
    season_h = float(np.mean(col["season_hours"]))
    drawn = sweep_drawn_phases(args)

    fig, axes, ax_note = panel_grid_with_notes(1, 1, 7.6, 5.0)
    ax = axes[0]
    unit = "" if as_hours else "%"
    for phase in drawn:
        pi = SWEEP_PHASES.index(phase)
        y = frac[:, slot, pi] * (season_h if as_hours else 100.0)
        # Endpoint values in the LEGEND rather than annotated on the axes: how
        # far the answer moves between the ends is the whole point of the
        # figure, but at the low end the three curves converge and on-axes
        # labels land on top of each other.
        ax.plot(lwp, y, color=PHASE_COLORS[phase], lw=2.2, marker="o",
                markersize=4,
                label=f"{MONTH_BAR_LABELS[phase]}:  {y[0]:,.1f}{unit} "
                      f"$\\rightarrow$ {y[-1]:,.1f}{unit}")
    _sweep_axes(ax, lwp, sweep["spacing"])
    ax.set_xlabel("minimum LWP [g m$^{-2}$]", fontsize=11)
    ax.set_ylabel("Hours per season\nper grid cell" if as_hours else
                  "Share of ALL cell-hours\nin the season [%]", fontsize=11)
    ax.legend(fontsize=9.5, framealpha=0.9, loc="best")

    what = "hours per season" if as_hours else "share of the season"
    suptitle_over_panels(
        fig,
        f"Cloud phase vs the minimum LWP threshold — {args.region}, "
        f"{series_label}\n{A.mode_label}"
        f"{', mean across seasons' if col['n_seasons'] > 1 else ''}"
        f"   |   {what}",
        1, fontsize=12.5)
    draw_notes(ax_note, sweep_note_lines(
        col, args,
        "y axis: hours per season = the season's share of cell-hours times the "
        "window length." if as_hours else
        "Denominator: ALL cell-hours in the season window, clear ones "
        "included."))

    if out_dir is not None:
        tag = "all" if surface_class in (None, "all") else str(surface_class)
        kind = "hours" if as_hours else "fraction"
        path = (Path(out_dir) / f"{args.region}_sweep_season_{kind}_{tag}_"
                                f"{A.tag}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig


# ----------------------------------------------------------------------------
# Liquid-CONTAINING hours vs the LWP threshold
# ----------------------------------------------------------------------------
# "Liquid-containing" is liquid-only PLUS mixed-phase: every overcast cell-hour
# that carries any liquid the threshold admits. It is the union of two of the
# three drawn categories, so it is immune to the reclassification argument in
# the notebook -- a scene moving from mixed to liquid-only, or back, does not
# change this number at all. Only a scene losing its liquid entirely does, which
# is what makes this the cleanest single curve to quote against --min-lwp.
SWEEP_LIQUID_CONTAINING: tuple[str, ...] = ("liquid", "mixed")

# Colour for the single-series (ARM site) figure. The by-class figure uses
# CLASS_COLORS so it matches the surface-class time-series figures exactly.
LIQUID_HOURS_COLOR = PHASE_COLORS["liquid"]


def sweep_liquid_containing_hours(A: Analysis) -> dict:
    """Average hours per season per grid cell that held a liquid-bearing cloud.

    Returns a dict with

    ``lwp``        (thr,)              the threshold axis, g m-2
    ``hours``      (thr, cls, season)  hours in EACH season, per grid cell
    ``mean``       (thr, cls)          mean of that over the seasons
    ``season_hours`` (season,)         each season's own window length

    The class axis is indexed by :func:`sweep_class_slot`, the same as
    ``sweep["season_fraction"]``.

    Computed per season and averaged afterwards, NOT as the stored mean
    fraction times a mean window length. ``season_fraction`` has already
    collapsed the season axis, and a common-year Oct-Mar season is 182 days
    against a leap year's 183 -- so the stored form cannot pair each season's
    fraction with its own window, and it cannot report a spread. Both come out
    of ``A.sec``, which still carries the full (thr, season, month, class,
    phase) accumulator, so this needs no reload.

    The weights are the same cell weights the rest of the module uses
    (cos-latitude by default), which makes the per-class number an
    area-weighted average over the cells of that class -- a "typical grid cell
    of this class". The ARM site slot is a single cell, so weighting is
    irrelevant there.
    """
    sec = A.sec
    if not sec["w_sweep"].size:
        raise ValueError(
            "no --min-lwp sweep was accumulated. It exists only in "
            "phase_mode='fraction'. Re-run prepare(phase_mode='fraction').")

    keep = list(A.keep_idx)
    ws = sec["w_sweep"][:, keep]                      # (thr, s, month, cls, ph)
    wss = sec["w_sweep_site"][:, keep]                # (thr, s, month, ph)
    all_cls = ws.sum(axis=3, keepdims=True)           # exact: the six partition
    ws_full = np.concatenate([ws, wss[:, :, :, None, :], all_cls], axis=3)

    # Numerator: pool the months, then add the two liquid-bearing phases.
    pi = [SWEEP_PHASES.index(p) for p in SWEEP_LIQUID_CONTAINING]
    num = ws_full.sum(axis=2)[..., pi].sum(axis=-1)   # (thr, s, cls)

    # Denominator: valid cell-hours, which do NOT depend on the threshold --
    # they are a property of the grid and the classification, not of --min-lwp.
    den_cls = sec["w_valid_month"][keep]              # (s, month, class7)
    order = [CLASS_CODES[n] for n in CLASS_ORDER]
    den = np.concatenate([
        den_cls[:, :, order],                                     # 5 classes
        den_cls[:, :, order].sum(axis=2, keepdims=True) * 0.0,    # unclassified
        den_cls[:, :, [sec["site_code"], sec["all_code"]]],       # site, all
    ], axis=2).sum(axis=1)                            # (s, cls)
    den[:, SWEEP_UNCLASSIFIED_SLOT] = np.maximum(
        den_cls[:, :, sec["all_code"]].sum(axis=1)
        - den_cls[:, :, order].sum(axis=(1, 2)), 0.0)

    season_h = season_window_hours(A.layout, keep)    # (s,) each season's own
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den[None] > 0,
                        num / np.where(den[None] > 0, den[None], 1.0),
                        np.nan)                       # (thr, s, cls)
    hours = frac * season_h[None, :, None]            # (thr, s, cls)
    hours = np.moveaxis(hours, 1, 2)                  # (thr, cls, s)
    return {
        "lwp": sec["sweep_lwp"],
        "spacing": sec["sweep_spacing"],
        "hours": hours,
        "mean": nanmean_quiet(hours, axis=2),         # (thr, cls)
        "season_hours": season_h,
    }


def liquid_hours_note_lines(A: Analysis, extra: str) -> list[str]:
    """Side note shared by the two liquid-containing-hours figures."""
    col, args = A.col, A.args
    pk = col["phase_kw"]
    lwp = col["sweep"]["lwp"]
    return [
        f"Season {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
        f"{args.season_end[0]:02d}-{args.season_end[1]:02d}",
        f"  ({window_label(col['season_hours'])} per season)",
        "",
        f"Cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}",
        "CWP = LWP + IWP",
        f"Liquid only: LWP/CWP $\\geq$ {pk['liquid_fraction_min']:g}",
        f"Ice only: IWP/CWP $\\geq$ {pk['ice_fraction_min']:g}",
        "Liquid-containing: neither,",
        "  i.e. liquid only + mixed",
        "",
        f"IWP floor held at {pk['min_iwp_g']:g} g m$^{{-2}}$",
        "x axis sweeps the LWP floor,",
        f"  {lwp[0]:g} to {lwp[-1]:g} g m$^{{-2}}$",
        f"  ({lwp.size} {col['sweep']['spacing']}ly spaced points)",
        "",
        wrap_note("Every point was classified in the same pass over the "
                  "archive, so the curve is built from identical cell-hours.",
                  34),
        "",
        wrap_note(extra, 34),
    ]


def fig_sweep_liquid_hours_site(A: Analysis, out_dir=None,
                                dpi: int | None = None,
                                surface_class: str = SITE_KEY,
                                min_max: bool = True):
    """Liquid-containing hours per season, ONE grid cell, against the LWP floor.

    Defaults to the ARM site cell (Utqiagvik). Each season's liquid-only and
    mixed-phase hours are summed, then the seasons are averaged; with
    ``min_max`` the band is the min-max across those seasons, which is the
    honest measure of how much of the y value is threshold and how much is
    year-to-year variability.
    """
    import matplotlib.pyplot as plt

    lh = sweep_liquid_containing_hours(A)
    args = A.args
    slot, series_label = sweep_class_slot(surface_class)
    lwp, y = lh["lwp"], lh["mean"][:, slot]
    per_season = lh["hours"][:, slot, :]              # (thr, season)

    fig, axes, ax_note = panel_grid_with_notes(1, 1, 7.6, 5.0)
    ax = axes[0]
    if min_max and per_season.shape[1] > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lo = np.nanmin(per_season, axis=1)
            hi = np.nanmax(per_season, axis=1)
        ax.fill_between(lwp, lo, hi, color=LIQUID_HOURS_COLOR, alpha=0.18,
                        linewidth=0,
                        label=f"min-max across {per_season.shape[1]} seasons")
    ax.plot(lwp, y, color=LIQUID_HOURS_COLOR, lw=2.4, marker="o", markersize=4,
            label=f"Liquid-containing:  {y[0]:,.0f} h "
                  f"$\\rightarrow$ {y[-1]:,.0f} h")
    _sweep_axes(ax, lwp, lh["spacing"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("minimum LWP [g m$^{-2}$]", fontsize=11)
    ax.set_ylabel("Liquid-containing cloud hours\nper season, per grid cell",
                  fontsize=11)
    ax.legend(fontsize=9.5, framealpha=0.9, loc="best")

    suptitle_over_panels(
        fig,
        f"Liquid-containing cloud hours vs the minimum LWP threshold — "
        f"{args.region}, {series_label}\n"
        f"liquid-only + mixed-phase, summed per season then averaged over "
        f"{len(A.used)} seasons",
        1, fontsize=12.5)
    draw_notes(ax_note, liquid_hours_note_lines(
        A, "ONE grid cell. Each season's own window length times that "
           "season's liquid-containing share, then averaged."))

    if out_dir is not None:
        path = (Path(out_dir) / f"{args.region}_sweep_liquid_hours_"
                                f"{surface_class}_{A.tag}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig


def fig_sweep_liquid_hours_by_class(A: Analysis, out_dir=None,
                                    dpi: int | None = None,
                                    include_site: bool = True,
                                    include_all: bool = False):
    """The same curve, one line per surface class, plus the ARM site cell.

    Colours are :data:`surface_classification.CLASS_COLORS`, and the site is
    black dashed, so this figure reads against the surface-class time series
    without a translation step.

    Each line is the average over the cells of that class (area-weighted, as
    everywhere else in the module) of the per-cell liquid-containing hours,
    averaged over the seasons -- i.e. "a typical grid cell of this class". A
    class holding less than ``--min-class-area`` of the domain gets its area
    printed in the legend, because a typical cell of a two-cell class is not a
    meaningful object.
    """
    import matplotlib.pyplot as plt

    lh = sweep_liquid_containing_hours(A)
    col, args = A.col, A.args
    lwp = lh["lwp"]

    series = [(name, CLASS_LABELS[name], CLASS_COLORS[name], "-", 2.2)
              for name in CLASS_ORDER]
    if include_all:
        series.append(("all", "All cells", "#444444", ":", 2.0))
    if include_site:
        series.append((SITE_KEY, SITE_LABEL, SITE_COLOR, "--", 1.9))

    fig, axes, ax_note = panel_grid_with_notes(1, 1, 7.6, 5.0)
    ax = axes[0]
    for name, label, color, ls, lw in series:
        slot, _ = sweep_class_slot(name)
        y = lh["mean"][:, slot]
        tail = ""
        if name in CLASS_CODES:
            share = col["area_pct"][CLASS_CODES[name]]
            if share < args.min_class_area:
                tail = f"  [{share:.2f}% of domain]"
        ax.plot(lwp, y, color=color, lw=lw, ls=ls, marker="o", markersize=3.4,
                solid_capstyle="round",
                label=f"{label}:  {y[0]:,.0f} $\\rightarrow$ {y[-1]:,.0f} h"
                      f"{tail}")
    _sweep_axes(ax, lwp, lh["spacing"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("minimum LWP [g m$^{-2}$]", fontsize=11)
    ax.set_ylabel("Liquid-containing cloud hours\nper season, per grid cell",
                  fontsize=11)
    ax.legend(fontsize=8.8, framealpha=0.9, loc="best")

    suptitle_over_panels(
        fig,
        f"Liquid-containing cloud hours vs the minimum LWP threshold — "
        f"{args.region}, by surface class\n"
        f"liquid-only + mixed-phase, summed per season then averaged over "
        f"{len(A.used)} seasons",
        1, fontsize=12.5)
    draw_notes(ax_note, liquid_hours_note_lines(
        A, "Averaged over the cells of the class: a TYPICAL cell of that "
           "class, not a class total."))

    if out_dir is not None:
        path = (Path(out_dir) / f"{args.region}_sweep_liquid_hours_by_class_"
                                f"{A.tag}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig


def print_liquid_hours_table(A: Analysis) -> None:
    """The two figures' numbers as text: hours at each end of the sweep."""
    lh = sweep_liquid_containing_hours(A)
    lwp = lh["lwp"]
    print(f"\n  Liquid-containing cloud hours per season per grid cell "
          f"(liquid-only + mixed), mean over {len(A.used)} seasons:")
    print(f"    {'series':<24}{f'{lwp[0]:g} g/m2':>12}"
          f"{f'{lwp[-1]:g} g/m2':>12}{'change':>10}")
    rows = [(CLASS_LABELS[n], n) for n in CLASS_ORDER]
    rows += [(SITE_LABEL, SITE_KEY), ("All cells", "all")]
    for label, name in rows:
        slot, _ = sweep_class_slot(name)
        y = lh["mean"][:, slot]
        print(f"    {label:<24}{y[0]:>12,.0f}{y[-1]:>12,.0f}"
              f"{100.0 * (y[-1] - y[0]) / y[0]:>9.1f}%")

# ----------------------------------------------------------------------------
# Minimum CLOUD DURATION, the ARM pipeline's other knob
# ----------------------------------------------------------------------------
# The ARM thermodynamic-cloud-phase product at Utqiagvik samples every 30 s, so
# a cloud can be timed: a run of consecutive liquid-bearing samples bounded by
# clear sky is an EVENT with a duration, and the pipeline can be told to ignore
# events shorter than some minimum. Raising that minimum removes short-lived
# cloud, and the liquid-containing hours fall.
#
# ERA5 samples every hour. THE MINIMUM RESOLVABLE DURATION IS THEREFORE ONE
# HOUR, which is already the full width of a 0-60 minute observational axis.
# The two sweeps do not overlap: ERA5 can only extend such a curve to the RIGHT
# of the observational one, and it cannot say anything at all about whether an
# hour ERA5 calls liquid-bearing was cloudy for 60 minutes or for 6. Every hour
# with LWP above the floor is credited with a full hour here -- that is an
# assumption of the archive's sampling, not a measurement, and it is the reason
# ERA5 and the ARM retrieval are not directly comparable at short durations.
#
# The 90 s clear-sky tolerance in the observational pipeline has no faithful
# ERA5 analogue either: the smallest gap ERA5 can express is a whole hour, forty
# times longer. ``gap_h`` exists so the choice is explicit rather than hidden,
# but the honest default is 0 -- no bridging.
DEFAULT_MAX_DURATION_H = 24
DURATION_COLOR = "#1f77b4"      # the observational figure's blue

# Hours in one ERA5 time step. Named so the run-length arithmetic below reads
# as hours rather than as indices; see HOURS_PER_STEP at the top of the module.
DURATION_STEP_H = HOURS_PER_STEP


def site_liquid_containing_series(A: Analysis, force: bool = False) -> dict:
    """Hour-by-hour liquid-containing flag for the single ARM-site cell.

    Returns

    ``times``     (n,) datetime64, the in-window steps, in order
    ``liquid``    (n,) bool, cloudy AND (liquid-only or mixed-phase)
    ``s_of``      (n,) int, index into ``A.used`` of the season each step is in
    ``step_ok``   (n-1,) bool, is step i+1 exactly one hour after step i

    This is a SECOND pass over the archive, because run lengths cannot be
    recovered from the accumulator: ``build_histograms`` folds every step into
    (season, month, class, phase) bins, and a run of six liquid hours is
    indistinguishable there from six isolated ones. Only one cell is read, so
    the pass is cheap; the result is cached on ``A`` and reused.

    The phase thresholds are the NOMINAL ones (``--min-lwp`` / ``--min-iwp``),
    not a sweep value -- duration is being varied here, not the LWP floor.
    """
    cached = getattr(A, "_site_series", None)
    if cached is not None and not force:
        return cached

    args, ds, pk = A.args, A.ds, A.col["phase_kw"]
    if pk["mode"] != "fraction":
        raise ValueError(
            "the duration sweep is defined for phase_mode='fraction', where "
            "liquid-containing is 'liquid only + mixed phase' under one LWP "
            "floor. Re-run prepare(phase_mode='fraction').")
    if parse_utc_hours(getattr(args, "utc_hours", None)):
        raise ValueError(
            "--utc-hours leaves gaps in the hourly record, so consecutive "
            "steps are no longer consecutive HOURS and a run length is not a "
            "duration. Re-run prepare() without it.")

    layout = A.layout
    dos, s_idx, in_window = layout["dos"], layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[list(A.keep_idx)] = True
    use_step = in_window & (s_idx >= 0) & wanted[np.clip(s_idx, 0, None)]

    site_mask, site_lat, site_lon = site_cell_mask(ds)
    read_vars = list(REQUIRED_VARS)
    liquid_var = getattr(args, "liquid_var", DEFAULT_LIQUID_VAR)
    if liquid_var not in read_vars:
        read_vars.append(liquid_var)
    if args.no_precip:
        read_vars += [v for v in PRECIP_SOURCE_VARS[args.precip_var]
                      if v not in read_vars]

    times_all = np.asarray(ds["valid_time"].values)
    liq_parts, time_parts, s_parts = [], [], []
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours,
                                      keep_mask=use_step):
        n_t = block.sizes["valid_time"]
        sl = slice(i0, i0 + n_t)
        keep = use_step[sl]
        if not keep.any():
            continue
        # One cell out of the block, AFTER the same masking the main pass uses,
        # so the two cannot drift apart in what counts as cloudy.
        tclw_g = block[liquid_var].values[keep][:, site_mask][:, 0] * 1000.0
        tciw_g = block["tciw"].values[keep][:, site_mask][:, 0] * 1000.0
        tcc = block["tcc"].values[keep][:, site_mask][:, 0]
        raining = precip_mask(block, keep, args)[:, site_mask][:, 0]

        valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw_g)
        cloudy = valid & (tcc >= args.min_cloud_fraction) & ~raining
        f = fraction_phase_masks(
            tclw_g, tciw_g, pk["liquid_fraction_min"], pk["ice_fraction_min"],
            pk["min_lwp_g"], pk["min_iwp_g"],
        )
        liq_parts.append(cloudy & (f["liquid"] | f["mixed"]))
        time_parts.append(times_all[sl][keep])
        s_parts.append(s_idx[sl][keep])

    times = np.concatenate(time_parts)
    liquid = np.concatenate(liq_parts)
    s_abs = np.concatenate(s_parts)
    order = np.argsort(times, kind="stable")
    times, liquid, s_abs = times[order], liquid[order], s_abs[order]

    # Season index remapped onto A.used, so the caller never has to know about
    # the seasons prepare() skipped.
    remap = {int(s): i for i, s in enumerate(A.keep_idx)}
    s_of = np.array([remap[int(s)] for s in s_abs], dtype=np.intp)

    step_h = (np.diff(times).astype("timedelta64[s]").astype(np.float64)
              / 3600.0)
    step_ok = np.isclose(step_h, DURATION_STEP_H) & (s_of[1:] == s_of[:-1])

    out = {
        "times": times, "liquid": liquid, "s_of": s_of, "step_ok": step_ok,
        "site_lat": site_lat, "site_lon": site_lon,
        # Steps inside a season that are NOT one hour after their predecessor:
        # holes in the archive. Every one of them truncates whatever run it
        # falls in, so a gappy season under-reports long events.
        "n_holes": int(np.count_nonzero(
            (s_of[1:] == s_of[:-1]) & ~np.isclose(step_h, DURATION_STEP_H))),
    }
    A._site_series = out
    return out


def liquid_events(liquid: np.ndarray, step_ok: np.ndarray,
                  gap_h: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Split a boolean hourly series into events.

    Returns ``(duration_h, liquid_h)``, one entry per event: how long the event
    lasted end to end, and how many of those hours were actually liquid-
    bearing. The two differ only when ``gap_h > 0`` bridges a short non-liquid
    break, and keeping them apart is what makes the bridge honest -- a bridged
    hour extends the event's DURATION, so it can help the event clear the
    threshold, but it is not counted as an hour of liquid cloud.

    A run is broken by any step that is not exactly one hour after its
    predecessor (``step_ok``), so a hole in the archive and a season boundary
    both end an event rather than silently joining across it.
    """
    n = liquid.size
    if n == 0:
        return np.empty(0), np.empty(0)
    cont = np.zeros(n, dtype=bool)                 # cont[i]: i-1 joins to i
    cont[1:] = liquid[1:] & liquid[:-1] & step_ok
    starts = np.flatnonzero(liquid & ~cont)
    ends = np.flatnonzero(liquid & ~np.append(cont[1:], False))
    if starts.size == 0:
        return np.empty(0), np.empty(0)
    run_h = (ends - starts + 1) * DURATION_STEP_H

    if gap_h <= 0:
        return run_h, run_h.copy()

    # Merge neighbouring runs separated by a bridgeable gap. A gap is
    # bridgeable only if it is short enough AND unbroken in time -- a hole in
    # the archive is not a "clear sky gap", it is an absence of information.
    unbroken = np.concatenate([[0], np.cumsum(~step_ok)])   # over step index
    dur, liq_h = [], []
    cur_start, cur_liq = starts[0], run_h[0]
    for j in range(1, starts.size):
        gap = (starts[j] - ends[j - 1] - 1) * DURATION_STEP_H
        # unbroken[i] counts broken transitions BEFORE index i, so the
        # difference over [ends[j-1], starts[j]] is the number of holes inside
        # the gap itself -- including the transition INTO starts[j], which an
        # end-exclusive slice would miss.
        contiguous = unbroken[starts[j]] == unbroken[ends[j - 1]]
        if gap <= gap_h and contiguous:
            cur_liq += run_h[j]
        else:
            dur.append((ends[j - 1] - cur_start + 1) * DURATION_STEP_H)
            liq_h.append(cur_liq)
            cur_start, cur_liq = starts[j], run_h[j]
    dur.append((ends[-1] - cur_start + 1) * DURATION_STEP_H)
    liq_h.append(cur_liq)
    return np.asarray(dur, dtype=float), np.asarray(liq_h, dtype=float)


def sweep_liquid_duration_hours(A: Analysis,
                                max_duration_h: int = DEFAULT_MAX_DURATION_H,
                                gap_h: float = 0.0) -> dict:
    """Liquid-containing hours at the ARM site vs a minimum-duration filter.

    Mirrors the ARM pipeline's duration filter: an event shorter than the
    threshold is discarded ENTIRELY, so the hours it held are removed from the
    total, exactly as a 3-minute cloud is dropped by a 10-minute filter.

    Returns

    ``duration``  (thr,)          the threshold axis, in HOURS, starting at 1
    ``hours``     (thr, season)   surviving liquid hours in each season
    ``mean``      (thr,)          mean of that over the seasons
    ``n_events``  (thr, season)   how many events survive

    The threshold axis is integer hours because ERA5 is hourly; a 30-minute
    threshold is not a finer question ERA5 answers badly, it is a question the
    archive cannot represent at all.
    """
    ser = site_liquid_containing_series(A)
    thr = np.arange(1, int(max_duration_h) + 1, dtype=float) * DURATION_STEP_H
    n_s = len(A.keep_idx)
    hours = np.zeros((thr.size, n_s))
    n_ev = np.zeros((thr.size, n_s), dtype=np.int64)

    for s in range(n_s):
        m = ser["s_of"] == s
        idx = np.flatnonzero(m)
        # step_ok is defined BETWEEN steps, so slice it on the interior pairs
        # of this season's block; the season boundary is already False in it.
        sub_ok = ser["step_ok"][idx[:-1]] if idx.size > 1 else np.empty(0, bool)
        dur, liq_h = liquid_events(ser["liquid"][m], sub_ok, gap_h)
        for k, t in enumerate(thr):
            keep = dur >= t
            hours[k, s] = float(liq_h[keep].sum())
            n_ev[k, s] = int(keep.sum())

    return {
        "duration": thr,
        "hours": hours,
        "mean": nanmean_quiet(hours, axis=1),
        "n_events": n_ev,
        "gap_h": float(gap_h),
        "n_holes": ser["n_holes"],
        "site_lat": ser["site_lat"], "site_lon": ser["site_lon"],
    }


def season_span_label(used) -> str:
    """'2022/23-2025/26' for a list of season START years.

    The end year is taken modulo 100 AFTER incrementing, so 1999 renders as
    1999/00 rather than 1999/100.
    """
    def one(y: int) -> str:
        return f"{int(y)}/{(int(y) + 1) % 100:02d}"
    return one(used[0]) if len(used) == 1 else f"{one(used[0])}-{one(used[-1])}"


def duration_ticks(thr: np.ndarray) -> np.ndarray:
    """Ticks for the duration axis that always include its left end.

    The axis starts at 1 h -- ERA5's shortest resolvable event -- and a default
    locator puts its first tick at 5, which hides exactly the number a reader
    needs to see.
    """
    lo, hi = float(thr[0]), float(thr[-1])
    step = max(1.0, round((hi - lo) / 6.0))
    ticks = np.arange(lo, hi + 1e-9, step)
    if ticks[-1] < hi - 1e-9:
        ticks = np.append(ticks, hi)
    return ticks

def fig_sweep_liquid_hours_duration(A: Analysis, out_dir=None,
                                    dpi: int | None = None,
                                    max_duration_h: int = DEFAULT_MAX_DURATION_H,
                                    gap_h: float = 0.0,
                                    with_lwp_axis: bool = True,
                                    min_max: bool = True):
    """Liquid-containing hours at the Barrow cell against BOTH quality knobs.

    The duration curve (blue, upper axis) is the ERA5 analogue of the ARM
    pipeline's minimum-cloud-duration filter. With ``with_lwp_axis`` the LWP
    sweep from :func:`fig_sweep_liquid_hours_site` is drawn on the same panel
    against the lower axis, which is the observational figure's layout: two
    quality thresholds, one y axis, so the cost of each is read off the same
    scale.

    THE TWO AXES ARE NOT COMPARABLE TO THE OBSERVATIONAL ONES IN RANGE. ERA5's
    duration axis starts at 1 h, where the ARM one ends; see the note at
    DEFAULT_MAX_DURATION_H.
    """
    import matplotlib.pyplot as plt

    dur = sweep_liquid_duration_hours(A, max_duration_h, gap_h)
    args, pk = A.args, A.col["phase_kw"]
    y = dur["mean"]
    per_season = dur["hours"]

    fig, axes, ax_note = panel_grid_with_notes(1, 1, 7.6, 5.2)
    ax = axes[0]

    # The no-cutoff reference: every liquid hour, however short-lived. This is
    # the number every other point on the figure is a reduction of.
    ax.axhline(y[0], color=DURATION_COLOR, ls="--", lw=1.2, alpha=0.7)
    ax.annotate(f"{y[0]:,.0f} h  (1 h steps, no duration cutoff)",
                xy=(0.985, y[0]), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=9, color=DURATION_COLOR)

    ax_top = ax.twiny() if with_lwp_axis else ax
    if with_lwp_axis:
        # Duration on the TOP axis and LWP on the bottom, matching the
        # observational figure's assignment of the two knobs to the two axes.
        lh = sweep_liquid_containing_hours(A)
        slot, _ = sweep_class_slot(SITE_KEY)
        lwp, y_lwp = lh["lwp"], lh["mean"][:, slot]
        ax.plot(lwp, y_lwp, color="black", lw=1.8, marker="o", markersize=5,
                markerfacecolor="none", markeredgewidth=1.3,
                label="LWP threshold (lower axis)")
        ax.set_xlim(lwp[0], lwp[-1])
        ax.set_xlabel("Minimum LWP threshold  [g m$^{-2}$]", fontsize=11)
        ax_top.set_xlim(dur["duration"][0], dur["duration"][-1])
        ax_top.set_xlabel("Minimum cloud duration threshold  (hours)",
                          fontsize=11, color=DURATION_COLOR, labelpad=8)
        # Tick the LEFT END explicitly. It is 1 h, not 0, and that is the whole
        # point of the figure -- an automatic locator starts at 5 and hides it.
        ax_top.set_xticks(duration_ticks(dur["duration"]))
        ax_top.tick_params(axis="x", colors=DURATION_COLOR, labelsize=9.5)
        for sp in ("top",):
            ax_top.spines[sp].set_color(DURATION_COLOR)
    else:
        ax.set_xlim(dur["duration"][0], dur["duration"][-1])
        ax.set_xticks(duration_ticks(dur["duration"]))
        ax.set_xlabel("Minimum cloud duration threshold  (hours)", fontsize=11)

    if min_max and per_season.shape[1] > 1:
        ax_top.fill_between(dur["duration"], per_season.min(axis=1),
                            per_season.max(axis=1), color=DURATION_COLOR,
                            alpha=0.12, linewidth=0, zorder=1,
                            label=f"min-max across {per_season.shape[1]} seasons")
    ax_top.plot(dur["duration"], y, color=DURATION_COLOR, lw=2.0, marker="s",
                markersize=5, markerfacecolor="none", markeredgewidth=1.4,
                label="Cloud duration threshold"
                      + (" (upper axis)" if with_lwp_axis else ""), zorder=3)

    ax.set_ylim(0, max(y.max(), y[0]) * 1.18)
    ax.set_ylabel("Liquid-containing cloud hours\nper season, per grid cell",
                  fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9.5)
    handles = ax.get_legend_handles_labels()
    if with_lwp_axis:
        h2 = ax_top.get_legend_handles_labels()
        handles = (handles[0] + h2[0], handles[1] + h2[1])
    ax.legend(*handles, fontsize=9.5, framealpha=0.9, loc="lower left")

    suptitle_over_panels(
        fig,
        f"Cost of a quality threshold: liquid-containing cloud hours at "
        f"{args.region}\nERA5, {len(A.used)} cold seasons "
        f"({season_span_label(A.used)}), "
        f"{args.season_start[0]:02d}-{args.season_start[1]:02d} to "
        f"{args.season_end[0]:02d}-{args.season_end[1]:02d}",
        1, fontsize=12.5)
    draw_notes(ax_note, duration_note_lines(A, dur))

    if out_dir is not None:
        kind = "duration_and_lwp" if with_lwp_axis else "duration"
        path = (Path(out_dir) / f"{args.region}_sweep_liquid_hours_{kind}_"
                                f"{A.tag}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi or args.dpi, bbox_inches="tight")
        print(f"  -> {path}")
    return fig


def duration_note_lines(A: Analysis, dur: dict) -> list[str]:
    """Side note for the duration figure."""
    args, pk = A.args, A.col["phase_kw"]
    return [
        f"Season {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
        f"{args.season_end[0]:02d}-{args.season_end[1]:02d}",
        f"  ({window_label(A.col['season_hours'])} per season)",
        f"ARM cell {dur['site_lat']:.2f} N, {dur['site_lon']:.2f} E",
        "",
        f"Cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}",
        f"Floors: LWP {pk['min_lwp_g']:g}, IWP "
        f"{pk['min_iwp_g']:g} g m$^{{-2}}$",
        f"Liquid only: LWP/CWP $\\geq$ {pk['liquid_fraction_min']:g}",
        "Liquid-containing: + mixed",
        "",
        wrap_note("An EVENT is a run of consecutive liquid-containing hours. "
                  "One below the threshold is dropped whole, hours and all.",
                  34),
        "",
        wrap_note(f"Clear-sky gap bridged: {dur['gap_h']:g} h. ERA5's smallest "
                  f"gap is 1 h, so the ARM pipeline's 90 s tolerance has no "
                  f"analogue.", 34),
        "",
        wrap_note("ERA5 is HOURLY: every hour above the floor counts as a full "
                  "hour, and nothing below 1 h resolves. The observational "
                  "0-60 min axis lies entirely LEFT of this one.", 34),
        (wrap_note(f"\n!! {dur['n_holes']} in-season gaps in the record; each "
                   f"truncates the event it falls in.", 34)
         if dur["n_holes"] else None),
    ]


def print_liquid_duration_table(A: Analysis,
                                max_duration_h: int = DEFAULT_MAX_DURATION_H,
                                gap_h: float = 0.0) -> None:
    """The duration figure's numbers, with the event counts behind them."""
    dur = sweep_liquid_duration_hours(A, max_duration_h, gap_h)
    y = dur["mean"]
    print(f"\n  Liquid-containing cloud hours per season at the ARM site cell, "
          f"mean over {len(A.used)} seasons,")
    print(f"  LWP floor {A.col['phase_kw']['min_lwp_g']:g} g m-2, "
          f"gap bridged {gap_h:g} h:")
    print(f"    {'min duration':>13}{'hours':>10}{'% of 1 h':>10}"
          f"{'events/season':>15}{'mean event':>12}")
    for k, t in enumerate(dur["duration"]):
        ne = dur["n_events"][k].mean()
        print(f"    {t:>10.0f} h {y[k]:>10,.0f}{100 * y[k] / y[0]:>9.1f}%"
              f"{ne:>15,.0f}{(y[k] / ne if ne else np.nan):>11.1f} h")

def sweep_note_lines(col: dict, args, denom: str) -> list[str]:
    """Side-note text for the sweep figures.

    ``denom`` is passed in rather than derived from a flag, because it differs
    along two axes at once -- monthly vs seasonal, fraction vs hours -- and a
    single boolean got it wrong for the seasonal figures.
    """
    pk = col["phase_kw"]
    lwp = col["sweep"]["lwp"]
    return [
        f"Season {args.season_start[0]:02d}-{args.season_start[1]:02d} to "
        f"{args.season_end[0]:02d}-{args.season_end[1]:02d}",
        f"  ({window_label(col['season_hours'])} per season)",
        "",
        f"Cloudy: tcc $\\geq$ {args.min_cloud_fraction:g}",
        "",
        "CWP = LWP + IWP",
        f"Liquid only: LWP/CWP $\\geq$ {pk['liquid_fraction_min']:g}",
        f"Ice only: IWP/CWP $\\geq$ {pk['ice_fraction_min']:g}",
        "Mixed phase: everything else",
        "",
        f"IWP floor held at {pk['min_iwp_g']:g} g m$^{{-2}}$",
        "x axis sweeps the LWP floor,",
        f"  {lwp[0]:g} to {lwp[-1]:g} g m$^{{-2}}$",
        f"  ({lwp.size} {col['sweep']['spacing']}ly spaced points)",
        "",
        wrap_note("Every point was classified in the same pass over the "
                  "archive, so the curves share identical cell-hours.", 34),
        "",
        wrap_note(denom, 34),
    ]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def add_phase_args(parser: argparse.ArgumentParser) -> None:
    """Cloud phase thresholds, all in g m-2.

    Four independent numbers, two per drawn category, so each can be defined on
    its own terms. The only constraint between them is
    ``--liquid-iwp-max <= --mixed-iwp-min``, which is what keeps the categories
    from overlapping; it is checked at run time and the error explains itself.
    """
    group = parser.add_argument_group(
        "cloud phase (all thresholds in g m-2)",
        "liquid only:  LWP > --liquid-lwp-min  AND  IWP < --liquid-iwp-max\n"
        "mixed phase:  LWP > --mixed-lwp-min   AND  IWP > --mixed-iwp-min\n"
        "Requires --liquid-iwp-max <= --mixed-iwp-min, or the two would "
        "overlap and hours would be counted twice.",
    )
    group.add_argument(
        "--liquid-lwp-min", type=float, default=DEFAULT_LIQUID_LWP_MIN,
        metavar="G",
        help=f"Liquid path a LIQUID-ONLY cloud must exceed (default "
             f"{DEFAULT_LIQUID_LWP_MIN:g}). Deliberately higher than the mixed "
             f"floor: the category claims a radiatively liquid deck, not a "
             f"trace.",
    )
    group.add_argument(
        "--liquid-iwp-max", type=float, default=DEFAULT_LIQUID_IWP_MAX,
        metavar="G",
        help=f"Ice path a LIQUID-ONLY cloud must stay below (default "
             f"{DEFAULT_LIQUID_IWP_MAX:g}). Must not exceed --mixed-iwp-min.",
    )
    group.add_argument(
        "--mixed-lwp-min", type=float, default=DEFAULT_MIXED_LWP_MIN,
        metavar="G",
        help=f"Liquid path a MIXED-PHASE cloud must exceed (default "
             f"{DEFAULT_MIXED_LWP_MIN:g}). Only has to show liquid is present.",
    )
    group.add_argument(
        "--mixed-iwp-min", type=float, default=DEFAULT_MIXED_IWP_MIN,
        metavar="G",
        help=f"Ice path a MIXED-PHASE cloud must exceed (default "
             f"{DEFAULT_MIXED_IWP_MIN:g}). Also the floor for the ice-only "
             f"column in the report.",
    )
    group.add_argument(
        "--ice-iwp-min", type=float, default=DEFAULT_ICE_IWP_MIN, metavar="G",
        help=f"Ice path an ICE-ONLY cloud must exceed (default "
             f"{DEFAULT_ICE_IWP_MIN:g}). Independent of --mixed-iwp-min: how "
             f"much ice makes a cloud an ice cloud and how much makes a liquid "
             f"cloud mixed are separate judgements.",
    )
    group.add_argument(
        "--lwp-max-ice", type=float, default=DEFAULT_LWP_MAX_ICE_G, metavar="G",
        help=f"LWP below which a cloudy scene counts as ice only (default "
             f"{DEFAULT_LWP_MAX_ICE_G:g}). Report only: an ice cloud has no "
             f"liquid water path to bin and is never drawn. Must sit below "
             f"both LWP floors.",
    )


def add_fraction_phase_args(parser: argparse.ArgumentParser) -> None:
    """Thresholds for ``--phase-mode fraction``. Ignored in absolute mode."""
    group = parser.add_argument_group(
        "cloud phase - fraction mode (--phase-mode fraction)",
        "CWP = LWP + IWP, counting only species above their minimum path.\n"
        "liquid only:  LWP/CWP >= --liquid-fraction-min\n"
        "ice only:     IWP/CWP >= --ice-fraction-min\n"
        "mixed phase:  everything else holding cloud water.\n"
        "Requires --liquid-fraction-min + --ice-fraction-min > 1, or "
        "liquid-only and ice-only would overlap.",
    )
    group.add_argument(
        "--liquid-fraction-min", type=float, default=DEFAULT_LIQUID_FRACTION,
        metavar="F",
        help=f"Share of CWP that must be liquid for a LIQUID-ONLY cloud "
             f"(default {DEFAULT_LIQUID_FRACTION:g}).",
    )
    group.add_argument(
        "--ice-fraction-min", type=float, default=DEFAULT_ICE_FRACTION,
        metavar="F",
        help=f"Share of CWP that must be ice for an ICE-ONLY cloud (default "
             f"{DEFAULT_ICE_FRACTION:g}).",
    )
    group.add_argument(
        "--min-lwp", type=float, default=DEFAULT_MIN_LWP, metavar="G",
        help=f"Liquid below this g m-2 is treated as absent -- it enters "
             f"neither CWP nor its own share (default {DEFAULT_MIN_LWP:g}). "
             f"Without it a scene holding 0.05 g m-2 of liquid and nothing "
             f"else is 100%% liquid by share.",
    )
    group.add_argument(
        "--min-iwp", type=float, default=DEFAULT_MIN_IWP, metavar="G",
        help=f"Ice below this g m-2 is treated as absent, the same way "
             f"(default {DEFAULT_MIN_IWP:g}).",
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
    parser.add_argument("--phase-mode", choices=PHASE_MODES,
                        default=DEFAULT_PHASE_MODE,
                        help="How the three cloud-phase categories are defined: "
                             "'absolute' by g m-2 thresholds on LWP and IWP, "
                             "'fraction' by each species' share of the cloud "
                             f"water path (default {DEFAULT_PHASE_MODE}). Each "
                             "mode reads only its own threshold group below.")
    add_phase_args(parser)
    add_fraction_phase_args(parser)
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
    parser.add_argument("--cell-weighting", choices=("area", "uniform"),
                        default="area",
                        help="How cells are weighted when averaging over a "
                             "surface class (default area = cos(latitude)). "
                             "'area' makes a class average per unit AREA; "
                             "'uniform' makes it per GRID CELL. The classes are "
                             "latitude-structured, so the two are not "
                             "interchangeable.")
    parser.add_argument("--utc-hours", type=parse_utc_hours, default=None,
                        metavar="SPEC",
                        help="Keep only these UTC hours: '7,8,9' or "
                             "'7-9,19-21'; default all 24. Narrows the SAMPLE, "
                             "not the season window, so hours-per-season stay "
                             "comparable. Its purpose is tcslw: that field is a "
                             "forecast initialised at 06/18 UTC, so its error "
                             "against the tclw analysis grows with lead, and "
                             "restricting to short-lead hours is the only way "
                             "to isolate the liquid definition from forecast "
                             "error.")
    parser.add_argument("--liquid-var", choices=LIQUID_VARS,
                        default=DEFAULT_LIQUID_VAR,
                        help="Which ERA5 field is treated as liquid water "
                             f"(default {DEFAULT_LIQUID_VAR}). 'tcslw' restricts "
                             "it to supercooled liquid. See the note at "
                             "LIQUID_VARS: the two are NOT nested in the "
                             "archived fields, so the switch is not a strict "
                             "narrowing.")
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
                             "the lower LWP floor, below which there is "
                             "nothing to draw). Set it lower to "
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
    parser.add_argument("--sweep-lwp-min", type=float,
                        default=DEFAULT_SWEEP_LWP_MIN, metavar="G",
                        help=f"Low end of the --min-lwp sweep, g m-2 (default "
                             f"{DEFAULT_SWEEP_LWP_MIN:g}). Fraction mode only.")
    parser.add_argument("--sweep-lwp-max", type=float,
                        default=DEFAULT_SWEEP_LWP_MAX, metavar="G",
                        help=f"High end of the sweep (default "
                             f"{DEFAULT_SWEEP_LWP_MAX:g}).")
    parser.add_argument("--sweep-points", type=int,
                        default=DEFAULT_SWEEP_POINTS, metavar="N",
                        help=f"Thresholds between them (default "
                             f"{DEFAULT_SWEEP_POINTS}). All are classified in "
                             f"one pass over the archive.")
    parser.add_argument("--sweep-spacing", choices=SWEEP_SPACINGS,
                        default=DEFAULT_SWEEP_SPACING,
                        help=f"How the sweep thresholds are laid out, and the "
                             f"scale of the resulting x axis (default "
                             f"{DEFAULT_SWEEP_SPACING}). Linear puts the "
                             f"resolution above 1 g m-2, where the phase split "
                             f"actually moves; log resolves the sub-0.1 region "
                             f"where the threshold meets ERA5's quantisation.")
    parser.add_argument("--monthly-class", nargs="+", default=["all"],
                        choices=["all", SITE_KEY] + list(CLASS_ORDER),
                        metavar="NAME",
                        help="Which cells the monthly phase bar chart averages "
                             "over, one figure each (default: all, the whole "
                             f"domain). Choose from: all, {SITE_KEY}, "
                             f"{', '.join(CLASS_ORDER)}.")
    parser.add_argument("--layout", type=parse_layout, default=DEFAULT_LAYOUT,
                        metavar="RxC",
                        help=f"Panel grid (default "
                             f"{DEFAULT_LAYOUT[0]}x{DEFAULT_LAYOUT[1]} for the "
                             "five classes plus the ARM site). Widened "
                             "automatically if too small.")
    parser.add_argument("--no-precip", action="store_true",
                        help="Drop precipitating scenes from the cloudy "
                             "population. Off by default.")
    parser.add_argument("--precip-var", choices=PRECIP_VARS,
                        default=DEFAULT_PRECIP_VAR,
                        help="What decides 'precipitating'. 'path' uses the "
                             "column rain+snow water content, 'rate' uses tp "
                             "against the literature 0.1 mm/hr. See the "
                             "calibration note in the source.")
    parser.add_argument("--precip-path-max", type=float,
                        default=DEFAULT_PRECIP_PATH_MAX_G, metavar="G",
                        help=f"g m-2 of tcrw+tcsw at or above which a scene "
                             f"counts as precipitating (default "
                             f"{DEFAULT_PRECIP_PATH_MAX_G:g}).")
    parser.add_argument("--precip-rate-max", type=float,
                        default=DEFAULT_PRECIP_RATE_MAX_MM_HR, metavar="MM_HR",
                        help=f"mm hr-1 of tp at or above which a scene counts "
                             f"as precipitating (default "
                             f"{DEFAULT_PRECIP_RATE_MAX_MM_HR:g}).")
    parser.add_argument("--residual-mode", choices=RESIDUAL_MODES,
                        default=DEFAULT_RESIDUAL_MODE,
                        help="Units of the lower panel on the three "
                             "ERA5-vs-observations figures (default "
                             f"{DEFAULT_RESIDUAL_MODE}). 'percent_diff' is "
                             "100*(ERA5-obs)/obs, normalised by the OBSERVED "
                             "value; 'hours' is the raw difference. Each figure "
                             "also takes residual_mode= to override this for "
                             "one call.")
    parser.add_argument("--show-digitization-uncert", action="store_true",
                        help="Draw the grey band showing how much uncertainty "
                             "reading Genie's published figures by eye "
                             "introduces, on the three ERA5-vs-observations "
                             "residual panels. OFF by default: it is a "
                             "statement about the digitisation, not about the "
                             "data, and it fills the panel. Each figure also "
                             "takes show_digitization_uncert= to override this "
                             "for one call.")
    parser.add_argument("--min-class-area", type=float,
                        default=DEFAULT_MIN_CLASS_AREA_PCT, metavar="PCT",
                        help="Stamp a warning on a panel whose class holds less "
                             f"than this %% of the domain (default "
                             f"{DEFAULT_MIN_CLASS_AREA_PCT:g}).")
    parser.add_argument("--block-hours", type=int, default=DEFAULT_BLOCK_HOURS,
                        metavar="N",
                        help=f"Time steps held in memory at once (default "
                             f"{DEFAULT_BLOCK_HOURS}).")
    parser.add_argument("--show-ice-only", action="store_true",
                        help="Draw the ice-only curve on the sweep figures "
                             "(default off). It is easy to misread: min_lwp is "
                             "the level below which liquid is declared absent, "
                             "not a bar a cloud must clear, so raising it can "
                             "only push scenes INTO ice-only, never out. See "
                             "'Reading these figures' in the sweep notebook "
                             "before turning this on.")
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

    # Validate every option that depends only on the arguments BEFORE opening
    # anything, so a bad threshold pair fails in milliseconds instead of after a
    # nine-minute read of the archive. It also avoids leaving a half-built
    # Dataset behind when the validation raises.
    phase_kw = resolve_phase_thresholds(args)
    # The lowest LWP either drawn category can admit. Below it the log axis has
    # nothing to show, by definition rather than by accident.
    log_min = (lowest_drawn_lwp(phase_kw) if args.lwp_log_min is None
               else args.lwp_log_min)
    edge_sets = {
        "linear": linear_bin_edges(args.lwp_lin_max, args.lwp_lin_bins),
        "log": log_bin_edges(log_min, args.lwp_log_max, args.lwp_log_bins),
    }

    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm_da = load_land_sea_mask(
        args.region, resolve_data_root(args.storage, args.data_root),
        args.mask_grid,
    )
    lsm = align_lsm_to_grid(lsm_da, ds)

    missing = sorted((set(REQUIRED_VARS) | {args.liquid_var})
                     - set(ds.data_vars))
    if missing:
        raise KeyError(f"dataset is missing {missing}. Re-download with "
                       f"--var-set recommended or extended")

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
    print(f"  Liquid     : {LIQUID_VAR_LABEL[args.liquid_var]}")
    print(f"  Cell weight: {getattr(args, 'cell_weighting', 'area')}"
          f"{' (cos latitude)' if getattr(args, 'cell_weighting', 'area') == 'area' else ' (per grid cell)'}")
    _uh = parse_utc_hours(getattr(args, "utc_hours", None))
    if _uh:
        print(f"  UTC hours  : {', '.join(f'{h:02d}' for h in _uh)}"
              f"   ({len(_uh)} of 24)")
    print(f"  Phase mode : {phase_kw['mode']}")
    print(f"  Categories : {phase_definition_label(phase_kw, mathtext=False)}")
    print(f"               {ice_definition_label(phase_kw, mathtext=False)}")
    print(f"  LWP bins   : linear 0-{args.lwp_lin_max:g} in "
          f"{args.lwp_lin_bins} | log {log_min:g}-"
          f"{args.lwp_log_max:g} in {args.lwp_log_bins}")

    layout = season_layout(ds, args)
    keep_idx, used, mode_label = select_seasons(layout, args)
    print(f"\n  Reading {len(used)} season(s): {used}")

    sweep_values = (sweep_lwp_values(args.sweep_lwp_min, args.sweep_lwp_max,
                                     args.sweep_points, args.sweep_spacing)
                    if phase_kw["mode"] == "fraction" else np.empty(0))
    if sweep_values.size:
        print(f"  LWP sweep  : {sweep_values[0]:g} to {sweep_values[-1]:g} "
              f"g m-2 in {sweep_values.size} {args.sweep_spacing}ly spaced "
              f"steps, all in one pass")
    sec = build_histograms(ds, lsm, args, layout, keep_idx, edge_sets, phase_kw,
                           sweep_values)
    if sec["n_unclassified"]:
        print(f"  !! {sec['n_unclassified']:,} unclassified cell-times; run "
              f"surface_classification.py for the breakdown.", file=sys.stderr)
    col = to_hours_per_season(sec, keep_idx, edge_sets, layout)
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

    print(f"\n  Hours per season per grid cell ({window_label(season_h)} in the "
          f"window),"
          f" mean over {len(A.used)} season(s):")
    print(f"    {'class':<22}{'mean area %':>12}{'cloudy':>9}{'liquid':>9}"
          f"{'mixed':>9}{'ice':>9}{'neither':>9}{'liq+mix %':>11}"
          f"{'med LWP liq':>13}{'med LWP mix':>13}")
    print("    " + "-" * 116)
    worst_residual = 0.0
    for code, label, _ in panel_order(col["site_code"]):
        liq = float(np.nansum(mean_lin[code, phase_i["liquid"]]))
        mix = float(np.nansum(mean_lin[code, phase_i["mixed"]]))
        ice = float(np.nansum(mean_lin[code, phase_i["ice"]]))
        non = float(np.nansum(mean_lin[code, phase_i["none"]]))
        cloudy = col["cloudy_hours"][code]
        # The four categories partition the cloudy hours by construction, so
        # this has to close. Tracked rather than trusted: a silent leak here
        # would be invisible on the figure, which only ever draws two of them.
        if np.isfinite(cloudy) and cloudy > 0:
            worst_residual = max(worst_residual,
                                 abs(liq + mix + ice + non - cloudy) / cloudy)
        pct = 100.0 * (liq + mix) / cloudy if cloudy > 0 else float("nan")
        area = col["area_pct"][code]
        area_s = "      1 cell" if code == col["site_code"] else f"{area:12.2f}"
        med_liq = col["median_lwp_g"][code, PHASE_STACK.index("liquid")]
        med_mix = col["median_lwp_g"][code, PHASE_STACK.index("mixed")]
        print(f"    {label:<22}{area_s}{cloudy:>9.0f}{liq:>9.0f}{mix:>9.0f}"
              f"{ice:>9.0f}{non:>9.0f}{pct:>11.1f}"
              f"{med_liq:>13.3g}{med_mix:>13.3g}")
    print("    " + "-" * 116)
    print("    'mean area %' is a time average: three of the five classes "
          "follow the ice edge and")
    print("    move through the season, so it is not a fixed property of the "
          "class.")
    print("    The two medians are over that class's liquid-only and "
          "mixed-phase hours separately,")
    print("    not over the pooled bars.")
    pk = col["phase_kw"]
    print("    liquid + mixed is the bar height drawn. The four category "
          "columns partition 'cloudy'")
    print(f"    exactly (max residual {100 * worst_residual:.2e}% of cloudy "
          f"hours across the classes).")
    if pk["mode"] == "fraction":
        # The fraction scheme is exhaustive over anything holding cloud water,
        # so 'neither' means one thing only and is worth stating plainly.
        print("    'neither' is every overcast hour with no cloud water above "
              "the minimum paths")
        print(f"    ({pk['min_lwp_g']:g} g m-2 liquid, {pk['min_iwp_g']:g} ice)."
              f" The three phases are exhaustive over the rest, so")
        print("    nothing else can land there.")
    else:
        print("    'neither' is every overcast hour matching no category: at "
              "these thresholds, mostly")
        print(f"    cloud holding between {pk['mixed_lwp_min_g']:g} and "
              f"{pk['liquid_lwp_min_g']:g} g m-2 of liquid with under "
              f"{pk['mixed_iwp_min_g']:g} g m-2 of ice --")
        print("    too thin for 'liquid only', too dry for 'mixed phase'. It "
              "is real cloud, not error.")
    if worst_residual > 1e-9:
        print(f"    !! the categories do not close to rounding "
              f"({100 * worst_residual:.3g}% residual); this is a bug, not a "
              f"threshold choice.", file=sys.stderr)

    degenerate = phase_split_warning(col)
    if degenerate:
        print(f"\n  !! {degenerate}.", file=sys.stderr)

    # The scale-up the module docstring warns about, made visible: how many
    # hours each season actually contributed against the window they are
    # reported over.
    sampled = sec["w_steps"][A.keep_idx]
    if sampled.size:
        print(f"\n  Hours present in the archive per season: "
              f"{sampled.min():,.0f} to {sampled.max():,.0f} of "
              f"{np.mean(season_h):,.0f}."
              f"  Bar heights are rates scaled to the full window.")

    if len(A.used) > 1:
        print("\n  Spread across seasons, liquid + mixed hours per season:")
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


# ----------------------------------------------------------------------------
# Season phase stack
# ----------------------------------------------------------------------------
# Stack order from the bottom, plus the residual. "none" is drawn last, in a
# neutral grey, so the bar genuinely reaches the cloudy total instead of
# stopping short of the annotation printed above it. In fraction mode it is
# exactly the overcast hours carrying no cloud water above the minimum paths,
# so it is usually a sliver; in absolute mode it can be substantial.
SEASON_STACK_ORDER: tuple[str, ...] = ("liquid", "mixed", "ice", "none")
NONE_COLOR = "#c8cdd2"


def _save_stack(fig, A, out_dir, stem, dpi, suffix: str = ""):
    """Write a season-stack figure, matching the naming the other figures use."""
    if out_dir is None:
        return fig
    path = Path(out_dir) / f"{A.args.region}_{stem}_{A.tag}{suffix}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def season_phase_hours(A: Analysis):
    """Cell-hours per phase, per season, for a TYPICAL cell of each class.

    Returns ``(season_labels, hours, cloudy, season_h)`` with ``hours`` shaped
    ``(n_season, n_class, n_phase)`` in ``SEASON_STACK_ORDER``.

    ``season_h`` is an ARRAY over seasons, not
    a scalar: a common-year Oct-Mar window is 182 days and a leap-year one 183,
    so "the season window" is not one number.

    Built by summing the already-accumulated LWP histogram over its bar axis,
    which means it inherits the module's normalisation exactly:

        hours = counts / (cell-hours the cell spent IN the class) * season_hours

    That matters, and a more obvious construction gets it wrong. Multiplying a
    monthly phase FRACTION by the month's calendar hours looks equivalent but is
    not: three of the five classes are defined by sea ice concentration, so a
    cell is open ocean in September and sea ice in February, and the fraction's
    denominator is only the hours the cell actually spent in that class. Scaling
    it by the whole month credits the class with hours it did not exist for. On
    this archive that discrepancy reached 1,595 h -- a third of the season.

    So a bar reads: "if a cell were in this class for the whole season, it would
    spend N hours under a cloud of this phase". Identical to the convention the
    per-class histogram panels and the printed report already use, which is what
    makes the totals here agree with theirs.
    """
    col = A.col
    per_season = col["hours"]["linear"]["per_season"]   # (s, class, phase, bar)
    idx = [PHASE_ORDER_ACC.index(p) for p in SEASON_STACK_ORDER]
    hours = np.nansum(per_season, axis=-1)[..., idx]    # (s, class, phase)
    cloudy = hours.sum(axis=-1)
    labels = [f"{y}/{(y + 1) % 100:02d}" for y in A.used]
    return labels, hours, cloudy, np.asarray(col["season_hours"], dtype=float)


def _stack_one_axis(ax, season_labels, hours, cloudy, season_h, label,
                    annotate=True, fontsize=8, ylim=None):
    """Draw one stacked-bar panel: seasons on x, phase hours stacked."""
    x = np.arange(len(season_labels))
    bottom = np.zeros(len(season_labels))
    drawn = []
    for pi, phase in enumerate(SEASON_STACK_ORDER):
        h = np.nan_to_num(hours[:, pi])
        if phase == "none" and h.max() <= 0:
            continue
        color = NONE_COLOR if phase == "none" else PHASE_COLORS[phase]
        lab = "no phase" if phase == "none" else PHASE_LABELS[phase].lower()
        ax.bar(x, h, width=0.68, bottom=bottom, color=color, label=lab,
               edgecolor="white", linewidth=0.4)
        bottom += h
        drawn.append(phase)

    # The y limit is set from the tallest bar across EVERY panel, not this one,
    # because the axes share a y scale: sizing each panel to its own maximum
    # would let the tallest class overflow into the row above, which is exactly
    # what the annotations then collide with.
    top = ylim if ylim is not None else max(bottom.max(), 1.0) * 1.28
    if annotate:
        for si_i, (xi, tot) in enumerate(zip(x, bottom)):
            if not np.isfinite(tot) or tot <= 0:
                continue
            ax.text(xi, tot + 0.015 * top,
                    f"{tot:,.0f} h\n"
                    f"({100.0 * tot / _sh(season_h, si_i):.1f}%)",
                    ha="center", va="bottom", fontsize=fontsize, linespacing=1.1)
    ax.set_ylim(0, top)

    ax.set_xticks(x)
    ax.set_xticklabels(season_labels, fontsize=fontsize + 0.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(label, fontsize=10)
    return drawn


def _sh(season_h, i):
    """One season's window length, whether given an array or a scalar."""
    a = np.asarray(season_h, dtype=float)
    return float(a if a.ndim == 0 else a[i])


def _stack_subtitle(A, season_h):
    pk = A.col["phase_kw"]
    return (f"cell-hours for a typical cell of each class   |   "
            f"season window {window_label(season_h)}   |   "
            f"{phase_definition_label(pk)}")


def fig_season_phase_stack(A: Analysis, out_dir=None, dpi: int | None = None):
    """Stacked phase hours per season -- one panel per surface class.

    Liquid on the bottom, mixed in the middle, ice on top, matching the order
    the caller asked for. Each bar's total is the overcast cell-hours a typical
    cell of that class saw in that season; the figure above each bar prints that
    total and, in parentheses, its share of the whole season window.

    The bar heights are per-cell hours, NOT summed over the class, so panels are
    directly comparable to the single ARM cell and to each other regardless of
    how many cells each class holds.
    """
    import matplotlib.pyplot as plt

    args = A.args
    labels, hours, cloudy, season_h = season_phase_hours(A)
    panels = panel_order(A.col["site_code"])
    n_r, n_c = DEFAULT_LAYOUT
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.3 * n_c, 4.3 * n_r),
                             sharey=True, constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.10)
    axes = np.atleast_1d(axes).ravel()

    # One headroom for every panel: 28% above the tallest bar anywhere, which is
    # what the two-line annotation needs without running into the row above.
    codes = [c for c, _, _ in panels]
    top = float(np.nanmax(cloudy[:, codes])) * 1.28

    drawn = []
    for ax, (code, label, is_site) in zip(axes, panels):
        if np.all(~np.isfinite(hours[:, code, :])) or cloudy[:, code].max() <= 0:
            ax.set_visible(False)
            continue
        d = _stack_one_axis(ax, labels, hours[:, code, :], cloudy[:, code],
                            season_h, label + ("  (1 cell)" if is_site else ""),
                            ylim=top)
        drawn = d or drawn
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for k in range(0, len(axes), n_c):
        if axes[k].get_visible():
            axes[k].set_ylabel("Cell-hours in season", fontsize=10)
    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Cloud phase hours by season and surface class \u2014 "
                 f"{args.region}\n{_stack_subtitle(A, season_h)}", fontsize=12)
    return _save_stack(fig, A, out_dir, "season_phase_stack", dpi)


def fig_season_phase_stack_site(A: Analysis, out_dir=None, dpi: int | None = None):
    """The same stack for the ARM cell alone, at full size.

    One grid cell, so no averaging over cells is involved -- this is what ERA5
    says the column over the facility did, season by season.
    """
    import matplotlib.pyplot as plt

    args = A.args
    labels, hours, cloudy, season_h = season_phase_hours(A)
    code = A.col["site_code"]
    fig, ax = plt.subplots(figsize=(1.5 + 1.25 * len(labels), 6.0),
                           constrained_layout=True)
    _stack_one_axis(ax, labels, hours[:, code, :], cloudy[:, code], season_h,
                    "", fontsize=9.5)
    ax.set_ylabel("Cell-hours in season", fontsize=11)
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.set_title(f"Cloud phase hours by season \u2014 {SITE_LABEL}\n"
                 f"{args.region}   |   season window "
                 f"{window_label(season_h)}   |   "
                 f"{phase_definition_label(A.col['phase_kw'])}",
                 fontsize=12, pad=10)
    return _save_stack(fig, A, out_dir, "season_phase_stack_site", dpi)


# ----------------------------------------------------------------------------
# Two-category view, for comparison against the ARM observations
# ----------------------------------------------------------------------------
# Colours taken from Genie's observational figure so the two can be read side by
# side without a mental translation: plain red for anything containing liquid,
# plain blue for ice-only. Matplotlib's named "red"/"blue", which is what her
# plot uses.
GENIE_LIQUID_COLOR = "red"
GENIE_ICE_COLOR = "blue"
GENIE_CLEAR_COLOR = "#c8cdd2"


def season_phase_binary(A: Analysis):
    """Collapse the three phases to ``liquid-containing`` and ``ice-only``.

    Liquid-containing is liquid-only PLUS mixed-phase: both are scenes a
    ground-based instrument would report as having liquid somewhere in the
    column. Ice-only is unchanged. The small "no phase" residual is folded into
    ice-only rather than dropped, so the two categories still sum to the
    overcast total -- it is overcast time carrying no cloud water above the
    minimum paths, which no instrument would call liquid.

    Returns ``(labels, liquid, ice, clear, season_h)``, each hours array shaped
    ``(n_season, n_class)``.
    """
    labels, hours, cloudy, season_h = season_phase_hours(A)
    i = {p: SEASON_STACK_ORDER.index(p) for p in SEASON_STACK_ORDER}
    liquid = hours[..., i["liquid"]] + hours[..., i["mixed"]]
    ice = hours[..., i["ice"]] + hours[..., i["none"]]
    clear = np.clip(np.asarray(season_h, dtype=float)[:, None]
                    - (liquid + ice), 0.0, None)
    return labels, liquid, ice, clear, season_h


def fig_season_phase_binary(A: Analysis, out_dir=None, dpi: int | None = None,
                            surface_class: str = "arm_site",
                            include_clear: bool = False):
    """Liquid-containing vs ice-only hours per season, in Genie's colours.

    ``include_clear`` adds a grey remainder so each bar spans the whole season
    window, matching the layout of the observational figure. It is OFF by
    default because the caller asked for two categories; turn it on when placing
    the two figures side by side, since her bars run to the full season and
    these otherwise stop at the overcast total.

    WHAT IS AND IS NOT COMPARABLE against the ARM figure:

      * The coloured hours ARE comparable. Both are hours per season in which a
        liquid-containing or ice-only cloud was overhead.
      * Her figure splits PRECIPITATING cases into their own lighter shades.
        This one applies no precipitation filter, so its liquid-containing bar
        corresponds to her red PLUS pink, and its ice-only bar to her blue PLUS
        light blue.
      * She has a hatched "missing / no data" category. ERA5 has no gaps, so
        there is no counterpart; a season short of hours here is scaled up to
        the nominal window instead (see the module docstring).
      * ERA5 is a 0.25 deg grid-box average sampled hourly; the ARM instruments
        see a point, far faster. Occupancy fractions compare; event durations
        do not.
    """
    import matplotlib.pyplot as plt

    args = A.args
    labels, liquid, ice, clear, season_h = season_phase_binary(A)
    code, series_label = resolve_series_code(A.col, surface_class)
    liq, ic, clr = liquid[:, code], ice[:, code], clear[:, code]

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(labels), 6.2),
                           constrained_layout=True)
    x = np.arange(len(labels))
    tot_cloud = liq + ic
    ax.bar(x, liq, width=0.68, color=GENIE_LIQUID_COLOR,
           label="liquid containing", edgecolor="white", linewidth=0.4)
    ax.bar(x, ic, width=0.68, bottom=liq, color=GENIE_ICE_COLOR,
           label="ice only", edgecolor="white", linewidth=0.4)
    top = tot_cloud
    if include_clear:
        ax.bar(x, clr, width=0.68, bottom=tot_cloud, color=GENIE_CLEAR_COLOR,
               label="clear / not overcast", edgecolor="white", linewidth=0.4)
        top = tot_cloud + clr

    headroom = float(np.nanmax(top)) * 1.20
    for si_i, (xi, cloud_h, bar_h) in enumerate(zip(x, tot_cloud, top)):
        if not np.isfinite(cloud_h) or cloud_h <= 0:
            continue
        ax.text(xi, bar_h + 0.012 * headroom,
                f"{cloud_h:,.0f} h\n"
                f"({100.0 * cloud_h / _sh(season_h, si_i):.1f}%)",
                ha="center", va="bottom", fontsize=9, linespacing=1.1)
    ax.set_ylim(0, headroom)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_ylabel("Hours per season", fontsize=11)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    note = ("annotation = overcast hours"
            if include_clear else "bar = overcast hours")
    ax.set_title(f"Liquid-containing vs ice-only cloud hours \u2014 "
                 f"{series_label}\n{args.region}   |   season window "
                 f"{window_label(season_h)}   |   {note}   |   "
                 f"{phase_definition_label(A.col['phase_kw'])}",
                 fontsize=11.5, pad=10)
    tag = "site" if surface_class == "arm_site" else str(surface_class)
    return _save_stack(fig, A, out_dir, f"season_phase_binary_{tag}", dpi)


# ----------------------------------------------------------------------------
# Side-by-side comparison against the ARM observations
# ----------------------------------------------------------------------------
DEFAULT_OBS_FILE = "genie_arm_seasonal_hours.txt"
OBS_COLUMNS = ("with_liquid", "ice_only", "liq_precip", "ice_precip",
               "clear_sky", "others", "missing")
OBS_LEGEND_MEANS = {"with_liquid": 1474, "ice_only": 1071, "liq_precip": 282,
                    "ice_precip": 255, "clear_sky": 898, "others": 36,
                    "missing": 339}

# How far the observation files can be wrong, in hours, as a property of WHERE
# THEY CAME FROM rather than of the figure drawing them.
#
# The seasonal file now holds Genie's exact numbers, verified cell by cell
# against her spreadsheet, so its band is ZERO -- turning the band on draws
# nothing for those figures, which is the correct behaviour rather than a bug.
# The monthly file is still read off a rendered chart and keeps its band.
#
# Set the seasonal value back above zero only if that file goes back to holding
# digitised values.
OBS_SEASONAL_UNCERTAINTY_H = 0.0
OBS_MONTHLY_UNCERTAINTY_H = 30.0
OBS_BAR_COLOR = "#d1495b"
# Genie's shades for the precipitating categories, so an all-sky comparison can
# show them as their own segments rather than folding them in invisibly.
GENIE_LIQUID_PRECIP_COLOR = "pink"
GENIE_ICE_PRECIP_COLOR = "lightblue"

ERA5_HATCH = "...."  # stippling that marks ERA5 bars apart from the solid obs bars
MISSING_HATCH = "///"  # diagonal stripes marking a bar built from incomplete obs

# Text sizes for fig_era5_vs_obs, exposed as function arguments so a notebook
# can bump them per call rather than editing the source.
DEFAULT_COMPARISON_LABEL_FONTSIZE = 13.0   # axis labels
DEFAULT_COMPARISON_TICK_FONTSIZE = 12.0    # tick labels on both axes
DEFAULT_COMPARISON_LEGEND_FONTSIZE = 11.5  # both panels' legends


def load_observations(path=DEFAULT_OBS_FILE, check: bool = True) -> dict:
    """Read the ARM seasonal hours table.

    Returns ``{"seasons": [...], <column>: array, ...}``. With ``check``, the
    per-season columns are averaged and compared against the record means the
    source figure's legend states; a departure of more than 100 h is reported.

    The file now holds Genie's exact numbers, so the check is a regression
    guard rather than a digitisation sanity test -- every column should agree
    with its legend value to within rounding, and anything else means the file
    has been edited or the wrong one is being read.
    """
    seasons, rows = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != len(OBS_COLUMNS) + 1:
            raise ValueError(f"{path}: expected {len(OBS_COLUMNS) + 1} fields, "
                             f"got {len(parts)} in {line!r}")
        seasons.append(int(parts[0]))
        rows.append([float(v) for v in parts[1:]])
    if not rows:
        raise ValueError(f"{path} holds no data rows")
    arr = np.asarray(rows)
    out = {"seasons": seasons}
    out.update({c: arr[:, i] for i, c in enumerate(OBS_COLUMNS)})

    if check:
        bad = [(c, arr[:, i].mean(), OBS_LEGEND_MEANS[c])
               for i, c in enumerate(OBS_COLUMNS)
               if abs(arr[:, i].mean() - OBS_LEGEND_MEANS[c]) > 100]
        for c, got, want in bad:
            print(f"  !! {path}: {c} averages {got:,.0f} h but the source "
                  f"figure's legend says {want:,.0f} h", file=sys.stderr)
    return out


def obs_binary(obs: dict, exclude_precip: bool):
    """Collapse the observation table to liquid-containing and ice-only.

    ``exclude_precip`` decides whether the precipitating categories are dropped
    or folded in, and it MUST match how the ERA5 side was built. That pairing is
    the whole point of the precipitation filter: with the filter on, ERA5's
    cloudy population excludes precipitating scenes, so it has to be compared
    against the observations' non-precipitating categories alone.
    """
    if exclude_precip:
        return obs["with_liquid"].copy(), obs["ice_only"].copy()
    return (obs["with_liquid"] + obs["liq_precip"],
            obs["ice_only"] + obs["ice_precip"])


# ----------------------------------------------------------------------------
# Shared furniture for the three ERA5-vs-observations comparison figures
# ----------------------------------------------------------------------------
# What the lower panel shows. 'hours' is the raw difference in hours; the
# default 'percent_diff' expresses it as a share of the OBSERVED value, which is
# what makes a 100 h miss on October's 500 h readable against the same 100 h on
# February's 200 h.
RESIDUAL_MODES: tuple[str, ...] = ("percent_diff", "hours")
DEFAULT_RESIDUAL_MODE = "percent_diff"

RESIDUAL_YLABEL: dict[str, str] = {
    "hours": "ERA5 $-$ observations [h]",
    "percent_diff": "100 $\\times$ (ERA5 $-$ obs) / obs   [%]",
}


def resolve_residual_mode(mode, args=None) -> str:
    """Pick the residual mode, falling back to the run's ``--residual-mode``.

    Passing ``None`` at the call site means "whatever the run was configured
    with", so a notebook can set it once in ``prepare()`` and have all three
    figures follow, while still being able to override one of them by hand.
    """
    if mode is None:
        mode = getattr(args, "residual_mode", DEFAULT_RESIDUAL_MODE)
    if mode not in RESIDUAL_MODES:
        raise ValueError(f"unknown residual mode {mode!r}; "
                         f"choose from {list(RESIDUAL_MODES)}")
    return mode


DEFAULT_SHOW_DIGITIZATION_UNCERT = False


def resolve_show_band(show, args=None) -> bool:
    """Whether to draw the digitisation band, falling back to the run's flag."""
    if show is None:
        show = getattr(args, "show_digitization_uncert",
                       DEFAULT_SHOW_DIGITIZATION_UNCERT)
    return bool(show)


def residual_values(era_h, obs_h, mode: str) -> np.ndarray:
    """ERA5 against the observations, in hours or as a percentage of the obs.

    The percentage is normalised by the OBSERVED value, so it reads as "ERA5 is
    N% high/low relative to what was measured". Bars with no observed hours
    return NaN rather than infinity, and matplotlib simply omits them.
    """
    era_h = np.asarray(era_h, dtype=float)
    obs_h = np.asarray(obs_h, dtype=float)
    if mode == "hours":
        return era_h - obs_h
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(obs_h > 0,
                        100.0 * (era_h - obs_h) / np.where(obs_h > 0, obs_h, 1.0),
                        np.nan)


def draw_residual_band(ax_r, mode: str, band_h: float, bars=(),
                       width: float = 0.0, show: bool = False):
    """The digitisation-uncertainty band, in whichever units the panel uses.

    ``show`` is OFF by default: the band is a statement about how well a
    published figure could be read by eye, not about the data, and it dominates
    the panel visually. Turn it on with ``--show-digitization-uncert`` when
    judging whether a residual is meaningful at all.

    In hours it is a constant strip, because the digitisation error is a fixed
    number of hours. As a percentage it is NOT constant: the same +/-band_h is a
    larger relative error on a small observed value, so it is drawn per bar from
    that bar's own denominator. ``bars`` is a sequence of
    ``(x positions, observed hours)`` pairs, one per drawn category.

    Returns the caption for the band, or an empty string when it is off. The
    caption is NOT drawn here: a residual bar can reach anywhere inside the
    panel, so the captions belong in a reserved strip under the axes, laid out
    by the caller alongside any other footnotes.
    """
    if not show or band_h <= 0:
        # band_h == 0 means the observations are exact, so there is nothing to
        # shade and nothing to caption. Silently drawing a zero-height bar and
        # a caption promising uncertainty would be worse than drawing nothing.
        return ""
    if mode == "hours":
        ax_r.axhspan(-band_h, band_h, color="0.85", zorder=0)
        note = (f"grey band: +/-{band_h:g} h, the digitisation uncertainty of "
                f"the observation file")
    else:
        for xs, obs_h in bars:
            obs_h = np.asarray(obs_h, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                half = np.where(obs_h > 0,
                                100.0 * band_h / np.where(obs_h > 0, obs_h, 1.0),
                                np.nan)
            ax_r.bar(xs, 2.0 * half, width=width, bottom=-half, color="0.85",
                     edgecolor="none", zorder=0)
        note = (f"grey band: the same +/-{band_h:g} h of digitisation "
                f"uncertainty, as a percentage of each observed value")
    return note


def draw_figure_footnotes(fig, notes, fontsize: float = 7.5) -> None:
    """Stack short captions in the strip below the axes, one per line.

    Figure coordinates rather than axes coordinates, so nothing plotted can
    land on top of them and two captions can never overlap each other.
    """
    notes = [n for n in notes if n]
    for k, note in enumerate(reversed(notes)):
        fig.text(0.01, 0.008 + 0.017 * k, note, ha="left", va="bottom",
                 fontsize=fontsize, color="0.35")


def threshold_box_lines(A: Analysis) -> list[str]:
    """The two threshold statements the comparison figures carry.

    Line 1 is the overcast gate: a scene counts as cloudy only where tcc is at
    or above ``--min-cloud-fraction``.

    Line 2 is the liquid-containing boundary, and in fraction mode it is set by
    ``--ice-fraction-min``, NOT by ``--liquid-fraction-min``. "Liquid
    containing" on these figures is liquid-only PLUS mixed-phase -- every
    water-bearing scene that is not ice-only -- and ice-only is
    ``IWP/CWP >= ice_fraction_min``. So the cut sits at

        LWP/(LWP + IWP) > 1 - ice_fraction_min

    i.e. GREATER than the complement, not less. ``--liquid-fraction-min`` only
    moves the liquid-only/mixed boundary, which this merge is blind to.
    """
    args = A.args
    pk = A.phase_kw
    lines = [f"min cloud fraction = {100.0 * args.min_cloud_fraction:g}%"]
    if pk["mode"] == "fraction":
        cut = 100.0 * (1.0 - pk["ice_fraction_min"])
        lines.append(f"liquid containing: LWP/(LWP+IWP) > {cut:g}%")
    else:
        floor = min(pk["liquid_lwp_min_g"], pk["mixed_lwp_min_g"])
        lines.append(f"liquid containing: LWP > {floor:g} g m$^{{-2}}$")
    return lines


def draw_threshold_box(ax, A: Analysis, loc: str = "upper right",
                       fontsize: float = 9.0):
    """Stamp the two defining thresholds onto a comparison panel.

    Both figures and their saved PNGs otherwise carry no record of which
    thresholds produced them, which is exactly the ambiguity that makes a stale
    re-run hard to spot.
    """
    place = {"upper right": (0.995, 0.995, "right", "top"),
             "upper left": (0.005, 0.995, "left", "top")}
    if loc not in place:
        raise ValueError(f"unknown loc {loc!r}; choose from {list(place)}")
    x, y, ha, va = place[loc]
    ax.text(x, y, "\n".join(threshold_box_lines(A)), transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize, linespacing=1.4, zorder=6,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="0.55", linewidth=0.8, alpha=0.94))


def _residual_stem(stem: str, mode: str) -> str:
    """Keep the two residual modes in separate files rather than overwriting."""
    return stem if mode == "hours" else f"{stem}_pct"


def _bar_total_label(ax, xi, total_h, liquid_h, season_h, top,
                     fontsize=8.0):
    """Two-line label above a bar: total hours, then the liquid-containing share.

    The share is of the SEASON WINDOW, not of the bar, so ERA5 and the
    observations are divided by the same number and the two labels can be read
    against each other directly. For an observation bar with missing hours that
    denominator is too generous -- the site was not watched for the whole window
    -- which is what the bar's diagonal-stripe hatch marks (see MISSING_HATCH).
    """
    if not np.isfinite(total_h) or total_h <= 0:
        return
    ax.text(xi, total_h + 0.015 * top,
            f"{total_h:,.0f} h\n{100.0 * liquid_h / season_h:.1f}%",
            ha="center", va="bottom", fontsize=fontsize, linespacing=1.15)


def fig_era5_vs_obs(A: Analysis, obs_path=DEFAULT_OBS_FILE, out_dir=None,
                    dpi: int | None = None, surface_class: str = "arm_site",
                    label_fontsize: float = DEFAULT_COMPARISON_LABEL_FONTSIZE,
                    tick_fontsize: float = DEFAULT_COMPARISON_TICK_FONTSIZE,
                    legend_fontsize: float = DEFAULT_COMPARISON_LEGEND_FONTSIZE,
                    residual_mode: str | None = None,
                    show_digitization_uncert: bool | None = None,
                    show_residual: bool = True):
    """ERA5 beside the ARM observations, with the residual underneath.

    Upper panel: two stacked bars per season -- ERA5 on the left, observations
    on the right -- each split into liquid-containing and ice-only. A box in the
    corner states the two thresholds that define the categories, so a saved
    figure records the run that produced it.

    Lower panel: ERA5 against the observations, one bar per category per
    season. ``residual_mode`` chooses the units: ``'percent_diff'`` (the
    default) shows it as a share of the observed value, ``'hours'`` as the raw
    difference. ``None`` follows the run's ``--residual-mode``. Set
    ``show_residual=False`` to drop this panel entirely and save the upper
    panel alone -- the saved file name gets a `` - no-residual-panel`` suffix
    so it does not overwrite the two-panel version.

    The precipitation filter decides which observation categories are used, so
    that the two sides mean the same thing. With ``--no-precip`` set, ERA5's
    cloudy population has precipitating scenes removed and the observations are
    compared on their non-precipitating categories alone; without it, both sides
    include precipitation. Getting that pairing wrong is worth more than any
    threshold choice in the filter.

    Seasons present on only one side are dropped, and how many is reported in
    the subtitle rather than left silent.

    ``label_fontsize``, ``tick_fontsize``, and ``legend_fontsize`` size the
    axis labels, the tick labels on both axes, and both panels' legends
    respectively -- tune per call rather than editing the source.
    """
    import matplotlib.pyplot as plt

    args = A.args
    residual_mode = resolve_residual_mode(residual_mode, args)
    show_band = resolve_show_band(show_digitization_uncert, args)
    obs = load_observations(obs_path)
    labels, liquid, ice, _clear, season_h = season_phase_binary(A)
    code, series_label = resolve_series_code(A.col, surface_class)

    era_years = [int(l.split("/")[0]) for l in labels]
    obs_liq_all, obs_ice_all = obs_binary(obs, exclude_precip=args.no_precip)
    obs_idx = {y: i for i, y in enumerate(obs["seasons"])}

    shared = [y for y in era_years if y in obs_idx]
    dropped = len(era_years) - len(shared)
    if not shared:
        raise ValueError("no season appears in both the ERA5 run and "
                         f"{obs_path}")
    ei = {y: i for i, y in enumerate(era_years)}
    e_liq = np.array([liquid[ei[y], code] for y in shared])
    e_ice = np.array([ice[ei[y], code] for y in shared])
    o_liq = np.array([obs_liq_all[obs_idx[y]] for y in shared])
    o_ice = np.array([obs_ice_all[obs_idx[y]] for y in shared])
    miss = np.array([obs["missing"][obs_idx[y]] for y in shared])
    # Window length season by season -- leap years are 24 h longer.
    s_h = np.array([_sh(season_h, ei[y]) for y in shared])

    x = np.arange(len(shared))
    w = 0.38
    if show_residual:
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(2.0 + 1.35 * len(shared), 9.0), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(2.0 + 1.35 * len(shared), 6.2))
        ax_r = None

    # ERA5 bars carry the same liquid/ice colors as the obs bars but stippled
    # rather than solid, so the two datasets no longer need a second, easily
    # confused color (a colored outline) layered on top of the phase colors.
    missing_mask = miss > 0.05 * s_h
    for off, (lq, ic, name, stippled) in (
            (-w / 2, (e_liq, e_ice, "ERA5", True)),
            (+w / 2, (o_liq, o_ice, "ARM obs", False))):
        hatch = ERA5_HATCH if stippled else None
        liq_bars = ax.bar(x + off, lq, width=w,
               color="white" if stippled else GENIE_LIQUID_COLOR,
               edgecolor=GENIE_LIQUID_COLOR if stippled else "none",
               linewidth=0.7, hatch=hatch)
        ice_bars = ax.bar(x + off, ic, width=w, bottom=lq,
               color="white" if stippled else GENIE_ICE_COLOR,
               edgecolor=GENIE_ICE_COLOR if stippled else "none",
               linewidth=0.7, hatch=hatch)
        # Observation bars built from an incomplete season get diagonal
        # stripes instead of a solid fill, so the gap is visible on the bar
        # itself rather than relegated to a footnote. ERA5 keeps its stippling
        # regardless -- it has no missing hours to flag.
        if not stippled:
            for xi in x[missing_mask]:
                for patch in (liq_bars[xi], ice_bars[xi]):
                    patch.set_hatch(MISSING_HATCH)
                    patch.set_edgecolor("white")
                    patch.set_linewidth(0.6)

    # Totals above every bar: hours on top, liquid-containing share of the
    # season window underneath. Seasons whose observations are substantially
    # incomplete are flagged by the bar's own stripe hatch rather than a
    # separate mark here -- the "how many hours missing" note lives in the
    # lower panel, where it does not compete with the totals for space.
    top = float(max((e_liq + e_ice).max(), (o_liq + o_ice).max())) * 1.40
    for xi in x:
        _bar_total_label(ax, xi - w / 2, e_liq[xi] + e_ice[xi], e_liq[xi],
                         s_h[xi], top, fontsize=label_fontsize - 5.0)
        _bar_total_label(ax, xi + w / 2, o_liq[xi] + o_ice[xi], o_liq[xi],
                         s_h[xi], top, fontsize=label_fontsize - 5.0)
    ax.set_ylim(0, top)
    ax.set_ylabel("Hours per season", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=GENIE_LIQUID_COLOR, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc=GENIE_ICE_COLOR, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.35", lw=0.8,
                      hatch=ERA5_HATCH),
        plt.Rectangle((0, 0), 1, 1, fc="0.55", ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="0.55", ec="white", lw=0.6,
                      hatch=MISSING_HATCH),
    ]
    ax.legend(handles, ["liquid containing", "ice only",
                        "ERA5 (left, stippled)", "ARM obs (right, solid)",
                        "missing observations"],
              fontsize=legend_fontsize, ncol=2, framealpha=0.9, loc="upper left")
    draw_threshold_box(ax, A, loc="upper right",
                       fontsize=legend_fontsize - 2.0)

    band_note = ""
    if show_residual:
        # ---- residual panel -------------------------------------------------
        d_liq = residual_values(e_liq, o_liq, residual_mode)
        d_ice = residual_values(e_ice, o_ice, residual_mode)
        # Band first, so the residual bars sit on top of it.
        band_note = draw_residual_band(
            ax_r, residual_mode, OBS_SEASONAL_UNCERTAINTY_H,
            bars=((x - w / 2, o_liq), (x + w / 2, o_ice)), width=w, show=show_band)
        liq_r_bars = ax_r.bar(x - w / 2, d_liq, width=w, color=GENIE_LIQUID_COLOR,
                 edgecolor="white", linewidth=0.5, label="liquid containing")
        ice_r_bars = ax_r.bar(x + w / 2, d_ice, width=w, color=GENIE_ICE_COLOR,
                 edgecolor="white", linewidth=0.5, label="ice only")
        ax_r.axhline(0.0, color="0.3", lw=1.0)

        # Stripe the same seasons flagged "missing" in the upper panel, so the
        # residual there is not read as if it were on equal footing with the
        # rest, and say how many hours are missing here rather than in the
        # upper panel.
        for xi in x[missing_mask]:
            liq_r_bars[xi].set_hatch(MISSING_HATCH)
            ice_r_bars[xi].set_hatch(MISSING_HATCH)
        if missing_mask.any():
            floor = (max(OBS_SEASONAL_UNCERTAINTY_H, 50.0)
                     if residual_mode == "hours" else 10.0)
            resid_span = float(max(np.nanmax(np.abs(d_liq)),
                                   np.nanmax(np.abs(d_ice)), floor))
            star_margin = 0.05 * resid_span
            for xi in x[missing_mask]:
                y_top = np.nanmax([d_liq[xi], d_ice[xi], 0.0]) + star_margin
                ax_r.text(xi, y_top, f"{miss[xi]:,.0f} h\nmissing",
                          ha="center", va="bottom", fontsize=11.0,
                          color="0.25", linespacing=1.15)

        ax_r.set_ylabel(RESIDUAL_YLABEL[residual_mode], fontsize=label_fontsize)
        ax_r.tick_params(axis="both", labelsize=tick_fontsize)
        ax_r.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax_r.set_axisbelow(True)
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)
        # No legend here: the colours repeat the upper panel's, and the
        # stripe hatch names itself.
        tick_ax = ax_r
    else:
        tick_ax = ax
    tick_ax.set_xticks(x)
    tick_ax.set_xticklabels([f"{y}/{(y + 1) % 100:02d}" for y in shared],
                            rotation=45, ha="right", fontsize=tick_fontsize)

    pair = ("precipitating scenes EXCLUDED from both sides" if args.no_precip
            else "precipitation INCLUDED on both sides")
    note = f"   |   {dropped} ERA5 season(s) not in the obs file" if dropped else ""
    fig.suptitle(f"ERA5 against ARM observations \u2014 {series_label}\n"
                 f"{args.region}   |   {pair}   |   {precip_label(args)}{note}",
                 fontsize=12.5, y=0.965)
    if show_residual:
        fig.subplots_adjust(top=0.90, bottom=0.135, left=0.09, right=0.985)
    else:
        fig.subplots_adjust(top=0.86, bottom=0.22, left=0.09, right=0.985)
    draw_figure_footnotes(fig, [band_note])
    tag = "noprecip" if args.no_precip else "allsky"
    suffix = "" if show_residual else " - no-residual-panel"
    return _save_stack(fig, A, out_dir,
                       _residual_stem(f"era5_vs_obs_{tag}", residual_mode), dpi,
                       suffix=suffix)


def fig_era5_vs_obs_allsky(A: Analysis, obs_path=DEFAULT_OBS_FILE, out_dir=None,
                           dpi: int | None = None, surface_class="arm_site",
                           label_fontsize=DEFAULT_COMPARISON_LABEL_FONTSIZE,
                           tick_fontsize=DEFAULT_COMPARISON_TICK_FONTSIZE,
                           legend_fontsize=DEFAULT_COMPARISON_LEGEND_FONTSIZE,
                           residual_mode: str | None = None,
                           show_digitization_uncert: bool | None = None,
                           show_residual: bool = True):
    """All-sky comparison: nothing filtered, and the obs precip split shown.

    The companion to :func:`fig_era5_vs_obs`. There, precipitating scenes are
    removed from ERA5 and the observations are compared on their
    non-precipitating categories alone. Here nothing is removed from either
    side, and the observation bar shows its precipitating categories as their
    own segments in Genie's shades, so the part of the comparison the filter
    would have cut is visible rather than folded in.

    ERA5 has no counterpart to that split -- the run it is given includes
    precipitation but does not separate it -- so its bar stays two segments.
    That asymmetry is the point of the figure: it shows how much of the
    observations' cloud time is precipitating, which is what the filtered
    version removes.

    Clear sky is not drawn, as asked; bars are cloud hours only.

    Requires an UNFILTERED ``A`` -- if the run had ``no_precip`` set, its cloud
    hours already exclude precipitation and the two sides would not correspond.

    ``residual_mode`` sets the lower panel's units; see :func:`fig_era5_vs_obs`.
    Set ``show_residual=False`` to drop that panel and save the upper panel
    alone, with a `` - no-residual-panel`` suffix on the file name.
    """
    import matplotlib.pyplot as plt

    args = A.args
    residual_mode = resolve_residual_mode(residual_mode, args)
    show_band = resolve_show_band(show_digitization_uncert, args)
    if args.no_precip:
        raise ValueError(
            "fig_era5_vs_obs_allsky needs a run with no_precip=False; this "
            "Analysis already has precipitating scenes removed, so its bars "
            "cannot be set against the observations' all-sky categories.")

    obs = load_observations(obs_path)
    labels, liquid, ice, _clear, season_h = season_phase_binary(A)
    code, series_label = resolve_series_code(A.col, surface_class)

    era_years = [int(l.split("/")[0]) for l in labels]
    obs_idx = {y: i for i, y in enumerate(obs["seasons"])}
    shared = [y for y in era_years if y in obs_idx]
    dropped = len(era_years) - len(shared)
    if not shared:
        raise ValueError(f"no season appears in both the run and {obs_path}")
    ei = {y: i for i, y in enumerate(era_years)}
    e_liq = np.array([liquid[ei[y], code] for y in shared])
    e_ice = np.array([ice[ei[y], code] for y in shared])
    o_liq = np.array([obs["with_liquid"][obs_idx[y]] for y in shared])
    o_liq_p = np.array([obs["liq_precip"][obs_idx[y]] for y in shared])
    o_ice = np.array([obs["ice_only"][obs_idx[y]] for y in shared])
    o_ice_p = np.array([obs["ice_precip"][obs_idx[y]] for y in shared])
    miss = np.array([obs["missing"][obs_idx[y]] for y in shared])
    s_h = np.array([_sh(season_h, ei[y]) for y in shared])

    x = np.arange(len(shared))
    w = 0.38
    if show_residual:
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(2.0 + 1.35 * len(shared), 9.0), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(2.0 + 1.35 * len(shared), 6.2))
        ax_r = None

    ax.bar(x - w / 2, e_liq, width=w, color="white",
           edgecolor=GENIE_LIQUID_COLOR, linewidth=0.7, hatch=ERA5_HATCH)
    ax.bar(x - w / 2, e_ice, width=w, bottom=e_liq, color="white",
           edgecolor=GENIE_ICE_COLOR, linewidth=0.7, hatch=ERA5_HATCH)

    # Observation stack, grouped BY PHASE rather than in Genie's original
    # order, so the liquid block and the ice block are each contiguous and can
    # be read against the two-segment ERA5 bar beside them.
    missing_mask = miss > 0.05 * s_h
    bottom = np.zeros(len(shared))
    for seg, colr in ((o_liq, GENIE_LIQUID_COLOR),
                      (o_liq_p, GENIE_LIQUID_PRECIP_COLOR),
                      (o_ice, GENIE_ICE_COLOR),
                      (o_ice_p, GENIE_ICE_PRECIP_COLOR)):
        seg_bars = ax.bar(x + w / 2, seg, width=w, bottom=bottom, color=colr,
               edgecolor="none")
        # Diagonal stripes over every observation segment in a season with
        # incomplete coverage, so the gap is visible on the bar itself. ERA5
        # keeps its stippling regardless -- it has no missing hours to flag.
        for xi in x[missing_mask]:
            seg_bars[xi].set_hatch(MISSING_HATCH)
            seg_bars[xi].set_edgecolor("white")
            seg_bars[xi].set_linewidth(0.6)
        bottom += seg

    e_tot, o_tot = e_liq + e_ice, bottom
    # More headroom than the two-segment figure: these bars are taller and the
    # legend shares the upper-left corner with the labels above them.
    top = float(max(e_tot.max(), o_tot.max())) * 1.52
    for xi in x:
        _bar_total_label(ax, xi - w / 2, e_tot[xi], e_liq[xi], s_h[xi], top,
                         fontsize=label_fontsize - 5.0)
        _bar_total_label(ax, xi + w / 2, o_tot[xi], o_liq[xi] + o_liq_p[xi],
                         s_h[xi], top, fontsize=label_fontsize - 5.0)
    ax.set_ylim(0, top)
    ax.set_ylabel("Hours per season", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in
               (GENIE_LIQUID_COLOR, GENIE_LIQUID_PRECIP_COLOR,
                GENIE_ICE_COLOR, GENIE_ICE_PRECIP_COLOR)]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.35", lw=0.8,
                                 hatch=ERA5_HATCH))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="0.55", ec="white", lw=0.6,
                                 hatch=MISSING_HATCH))
    ax.legend(handles,
              ["liquid containing", "liquid containing (precip)",
               "ice only", "ice only (precip)", "ERA5 (left, stippled)",
               "missing observations"],
              fontsize=legend_fontsize, ncol=2, framealpha=0.9, loc="upper left")
    draw_threshold_box(ax, A, loc="upper right",
                       fontsize=legend_fontsize - 2.0)

    band_note = ""
    if show_residual:
        # ---- residual panel, against the obs totals INCLUDING precip -------
        o_liq_tot, o_ice_tot = o_liq + o_liq_p, o_ice + o_ice_p
        d_liq = residual_values(e_liq, o_liq_tot, residual_mode)
        d_ice = residual_values(e_ice, o_ice_tot, residual_mode)
        band_note = draw_residual_band(
            ax_r, residual_mode, OBS_SEASONAL_UNCERTAINTY_H,
            bars=((x - w / 2, o_liq_tot), (x + w / 2, o_ice_tot)), width=w,
            show=show_band)
        liq_r_bars = ax_r.bar(x - w / 2, d_liq, width=w, color=GENIE_LIQUID_COLOR,
                 edgecolor="white", linewidth=0.5)
        ice_r_bars = ax_r.bar(x + w / 2, d_ice, width=w, color=GENIE_ICE_COLOR,
                 edgecolor="white", linewidth=0.5)
        ax_r.axhline(0.0, color="0.3", lw=1.0)
        for xi in x[missing_mask]:
            liq_r_bars[xi].set_hatch(MISSING_HATCH)
            ice_r_bars[xi].set_hatch(MISSING_HATCH)
        if missing_mask.any():
            floor = (max(OBS_SEASONAL_UNCERTAINTY_H, 50.0)
                     if residual_mode == "hours" else 10.0)
            span = float(max(np.nanmax(np.abs(d_liq)), np.nanmax(np.abs(d_ice)),
                             floor))
            for xi in x[missing_mask]:
                ax_r.text(xi, np.nanmax([d_liq[xi], d_ice[xi], 0.0]) + 0.05 * span,
                          f"{miss[xi]:,.0f} h\nmissing", ha="center",
                          va="bottom", fontsize=11.0, color="0.25", linespacing=1.15)
        ax_r.set_ylabel(RESIDUAL_YLABEL[residual_mode], fontsize=label_fontsize)
        ax_r.tick_params(axis="both", labelsize=tick_fontsize)
        ax_r.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax_r.set_axisbelow(True)
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)
        tick_ax = ax_r
    else:
        tick_ax = ax
    tick_ax.set_xticks(x)
    tick_ax.set_xticklabels([f"{y}/{(y + 1) % 100:02d}" for y in shared],
                            rotation=45, ha="right", fontsize=tick_fontsize)

    note = f"   |   {dropped} ERA5 season(s) not in the obs file" if dropped else ""
    fig.suptitle(f"ERA5 against ARM observations, ALL SKY \u2014 {series_label}"
                 f"\n{args.region}   |   nothing filtered; the observations' "
                 f"precipitating categories are shown separately{note}",
                 fontsize=12.5, y=0.965)
    if show_residual:
        fig.subplots_adjust(top=0.90, bottom=0.135, left=0.09, right=0.985)
    else:
        fig.subplots_adjust(top=0.86, bottom=0.22, left=0.09, right=0.985)
    draw_figure_footnotes(fig, [band_note])
    suffix = "" if show_residual else " - no-residual-panel"
    return _save_stack(fig, A, out_dir,
                       _residual_stem("era5_vs_obs_allsky_split", residual_mode),
                       dpi, suffix=suffix)


# ----------------------------------------------------------------------------
# Months Genie drops from the observational record, and why ERA5 must drop them
# ----------------------------------------------------------------------------
# These calendar months are EXCLUDED FROM HER MONTHLY FIGURE because the ARM
# instruments were degraded and the retrieved cloud hours are anomalously low --
# an instrumentation artefact, not Arctic weather. They are therefore not in the
# means stored in genie_arm_monthly_hours.txt.
#
# ERA5 has no such gaps, so leaving these months in on the ERA5 side means the
# two monthly means are averages over DIFFERENT SAMPLES OF SEASONS, and the
# residual would carry that sampling difference on top of any real model bias.
# October is the clearest case: dropping Oct 2020 and Oct 2023 removes two of
# eleven seasons from the observed mean but none from ERA5's unless this list is
# applied.
#
# Stated as CALENDAR (year, month), which is how they were reported. The season
# each belongs to is derived from the run's own season window, so Jan 2020 lands
# in the 2019/20 season and Oct 2020 in 2020/21 without either being written
# down twice.
#
# SCOPE: the MONTHLY comparison only (section 7). The seasonal figures are not
# affected -- her seasonal figure keeps these months and reports the shortfall
# through its own "missing hours" category instead, which those figures already
# mark with an asterisk.
#
# If Genie supplies the real CSVs, or revises which months she rejects, this
# tuple is the single place to change.
GENIE_EXCLUDED_MONTHS: tuple[tuple[int, int], ...] = (
    (2020, 1),      # Jan 2020
    (2020, 10),     # Oct 2020
    (2020, 11),     # Nov 2020
    (2020, 12),     # Dec 2020
    (2021, 11),     # Nov 2021
    (2023, 2),      # Feb 2023
    (2023, 10),     # Oct 2023
)


def season_year_of(calendar_year: int, month: int, season_start_month: int,
                   wraps: bool) -> int:
    """Season START year holding a given calendar month.

    For a window that wraps the new year (Oct-Mar), a month at or after the
    start month belongs to the season beginning that calendar year, and a month
    before it belongs to the season that began the PREVIOUS year -- so Jan 2020
    is the 2019/20 season. For a window inside one year the two coincide.
    """
    if wraps and month < season_start_month:
        return calendar_year - 1
    return calendar_year


def excluded_month_mask(seasons, months, args,
                        exclude_months=GENIE_EXCLUDED_MONTHS):
    """Boolean ``(n_season, n_month)``: True where the month is excluded.

    Also returns the ``(calendar_year, month)`` pairs that actually landed
    inside the run, so a caller can say how many were dropped rather than
    silently averaging over fewer seasons. A pair outside the run's years or
    outside its season window is not an error -- it simply matches nothing.
    """
    seasons = list(seasons)
    months = list(months)
    s_of = {y: i for i, y in enumerate(seasons)}
    m_of = {m: j for j, m in enumerate(months)}
    start_month = args.season_start[0]
    wraps = tuple(args.season_end) < tuple(args.season_start)

    mask = np.zeros((len(seasons), len(months)), dtype=bool)
    hit = []
    for cal_year, month in exclude_months:
        s_year = season_year_of(cal_year, month, start_month, wraps)
        if s_year in s_of and month in m_of:
            mask[s_of[s_year], m_of[month]] = True
            hit.append((cal_year, month))
    return mask, hit


def monthly_phase_binary(A: Analysis, surface_class: str = "arm_site",
                         exclude_months=GENIE_EXCLUDED_MONTHS):
    """Monthly liquid-containing and ice-only hours, per season.

    Returns ``(months, liq, ice, month_h)`` with ``liq``/``ice`` shaped
    ``(n_season, n_month)`` in hours.

    ONLY VALID FOR A SINGLE, TIME-INVARIANT SERIES, which is why it defaults to
    the ARM cell and rejects anything else. The monthly fractions are
    normalised by the cell-hours the cell spent IN its class, so multiplying by
    the month's calendar hours is only the same thing when that membership does
    not move. It does not for one fixed cell; it does for open ocean and sea
    ice, which swap through the season, and the seasonal figures use a
    different route for exactly that reason.

    ``exclude_months`` blanks individual season-months to NaN so they drop out
    of any nanmean over seasons. It defaults to :data:`GENIE_EXCLUDED_MONTHS`,
    the months her instruments could not measure properly -- see the note on
    that constant for why matching them matters. Pass ``()`` for the unmasked
    numbers.
    """
    if surface_class not in ("arm_site", "all"):
        raise ValueError(
            "monthly_phase_binary is only valid for a series whose membership "
            "does not move through the season -- pass 'arm_site' (one cell) or "
            "'all' (the whole domain). Class membership migrates with the ice "
            "edge, so a monthly fraction times calendar hours would credit a "
            "class with hours it did not exist for.")
    col = A.col
    code, _ = resolve_series_code(col, surface_class)
    frac = col["month_fraction"]["per_season"]      # (s, month, class, phase)
    # Per SEASON as well as per month: February is 28 days in a common year and
    # 29 in a leap year, so one shared row of month lengths would scale every
    # common-year February up by 1/28. The shared-calendar form is kept only as
    # the fallback and for the axis furniture.
    month_h2 = col.get("month_hours")               # (season, month) or None
    month_h = month_window_hours(A.sec["slots"])    # (month,) nominal
    scale = month_h[None, :] if month_h2 is None else month_h2
    i = {p: PHASE_ORDER_ACC.index(p) for p in PHASE_ORDER_ACC}
    liq = (frac[:, :, code, i["liquid"]] + frac[:, :, code, i["mixed"]]) * scale
    ice = (frac[:, :, code, i["ice"]] + frac[:, :, code, i["none"]]) * scale
    if exclude_months:
        drop, _hit = excluded_month_mask(A.used, col["months"], A.args,
                                         exclude_months)
        liq = np.where(drop, np.nan, liq)
        ice = np.where(drop, np.nan, ice)
    if month_h2 is not None:
        month_h = np.asarray(month_h2, dtype=float).mean(axis=0)
    return col["months"], liq, ice, month_h


DEFAULT_MONTHLY_OBS_FILE = "genie_arm_monthly_hours.txt"


def load_monthly_observations(path=DEFAULT_MONTHLY_OBS_FILE) -> dict:
    """Read the ARM monthly-mean table: month -> (with_liquid, ice_only) hours.

    Both categories INCLUDE precipitating cases, so the matching ERA5 run is an
    unfiltered one. Returns ``{"months": [...], "with_liquid": arr,
    "ice_only": arr}``.
    """
    months, liq, ice = [], [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}: expected 3 fields, got {len(parts)} "
                             f"in {line!r}")
        months.append(int(parts[0]))
        liq.append(float(parts[1]))
        ice.append(float(parts[2]))
    if not months:
        raise ValueError(f"{path} holds no data rows")
    return {"months": months, "with_liquid": np.asarray(liq),
            "ice_only": np.asarray(ice)}


def fig_monthly_era5_vs_obs(A: Analysis, obs_path=DEFAULT_MONTHLY_OBS_FILE,
                            out_dir=None, dpi: int | None = None,
                            surface_class: str = "arm_site",
                            label_fontsize=DEFAULT_COMPARISON_LABEL_FONTSIZE,
                            tick_fontsize=DEFAULT_COMPARISON_TICK_FONTSIZE,
                            legend_fontsize=DEFAULT_COMPARISON_LEGEND_FONTSIZE,
                            residual_mode: str | None = None,
                            exclude_months=GENIE_EXCLUDED_MONTHS,
                            show_digitization_uncert: bool | None = None,
                            show_residual: bool = True):
    """Monthly mean cloud hours, ERA5 beside the observations.

    Upper panel: two stacked bars per month -- ERA5 stippled on the left,
    observations solid on the right -- each split into liquid-containing and
    ice-only, averaged over seasons. The whisker on ERA5's liquid segment is
    +/- one standard deviation ACROSS SEASONS, so it is interannual variability,
    not uncertainty in the mean. There is no counterpart for the observations:
    the file holds means only.

    Lower panel: ERA5 against the observations, per category.
    ``residual_mode`` sets its units; see :func:`fig_era5_vs_obs`.

    The observation file's categories INCLUDE precipitation, so an unfiltered
    ERA5 run is the matching one; passing a filtered ``A`` raises rather than
    quietly comparing different populations.

    ``exclude_months`` drops the season-months her instruments could not
    measure, so both sides average over the same seasons; it defaults to
    :data:`GENIE_EXCLUDED_MONTHS` and the count is stated on the figure. Pass
    ``()`` to compare without it.

    Every number here is gated on ``--min-cloud-fraction`` the same way the
    seasonal figures are: ``monthly_phase_binary`` reads ``month_fraction``,
    which is accumulated from ``w_phase_month``, which is masked by ``cloudy``.
    Lowering the gate raises these bars. The threshold box in the corner is
    there so a figure that did NOT move can be told apart from a stale one.

    Set ``show_residual=False`` to drop the lower panel and save the upper
    panel alone, with a `` - no-residual-panel`` suffix on the file name.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    args = A.args
    residual_mode = resolve_residual_mode(residual_mode, args)
    show_band = resolve_show_band(show_digitization_uncert, args)
    if args.no_precip:
        raise ValueError(
            "the monthly observation file includes precipitating cases, so "
            "this figure needs a run with no_precip=False; the Analysis given "
            "has precipitating scenes removed and the two would not correspond.")

    obs = load_monthly_observations(obs_path)
    months, liq, ice, month_h = monthly_phase_binary(A, surface_class,
                                                    exclude_months)
    _code, series_label = resolve_series_code(A.col, surface_class)
    _drop, dropped_months = excluded_month_mask(A.used, months, args,
                                                exclude_months or ())

    o_idx = {m: i for i, m in enumerate(obs["months"])}
    shared = [m for m in months if m in o_idx]
    if not shared:
        raise ValueError(f"no month appears in both the run and {obs_path}")
    mi = {m: i for i, m in enumerate(months)}
    e_liq = np.array([nanmean_quiet(liq[:, mi[m]]) for m in shared])
    e_ice = np.array([nanmean_quiet(ice[:, mi[m]]) for m in shared])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        e_liq_sd = np.array([np.nanstd(liq[:, mi[m]]) for m in shared])
    o_liq = np.array([obs["with_liquid"][o_idx[m]] for m in shared])
    o_ice = np.array([obs["ice_only"][o_idx[m]] for m in shared])
    m_hours = np.array([month_h[mi[m]] for m in shared], dtype=float)
    # Excluded months leave different bars averaged over different numbers of
    # seasons, so the legend states the range instead of one count.
    n_per_month = np.array([int(np.isfinite(liq[:, mi[m]]).sum())
                            for m in shared])
    n_lo, n_hi = int(n_per_month.min()), int(n_per_month.max())
    n_txt = f"{n_hi}" if n_lo == n_hi else f"{n_lo}\u2013{n_hi}"

    x = np.arange(len(shared))
    w = 0.38
    if show_residual:
        fig, (ax, ax_r) = plt.subplots(
            2, 1, figsize=(2.0 + 1.6 * len(shared), 9.0), sharex=True,
            gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(2.0 + 1.6 * len(shared), 6.2))
        ax_r = None

    ax.bar(x - w / 2, e_liq, width=w, color="white",
           edgecolor=GENIE_LIQUID_COLOR, linewidth=0.7, hatch=ERA5_HATCH)
    ax.bar(x - w / 2, e_ice, width=w, bottom=e_liq, color="white",
           edgecolor=GENIE_ICE_COLOR, linewidth=0.7, hatch=ERA5_HATCH)
    ax.bar(x + w / 2, o_liq, width=w, color=GENIE_LIQUID_COLOR, edgecolor="none")
    ax.bar(x + w / 2, o_ice, width=w, bottom=o_liq, color=GENIE_ICE_COLOR,
           edgecolor="none")
    # Whisker where ERA5's liquid segment ends, so it reads as that segment's
    # spread rather than the whole stack's.
    ax.errorbar(x - w / 2, e_liq, yerr=e_liq_sd, fmt="none", ecolor="0.15",
                elinewidth=1.5, capsize=5, capthick=1.5, zorder=5)

    e_tot, o_tot = e_liq + e_ice, o_liq + o_ice
    top = float(max((e_tot + e_liq_sd).max(), o_tot.max())) * 1.45
    for xi in x:
        _bar_total_label(ax, xi - w / 2, e_tot[xi], e_liq[xi], m_hours[xi], top,
                         fontsize=label_fontsize - 4.5)
        _bar_total_label(ax, xi + w / 2, o_tot[xi], o_liq[xi], m_hours[xi], top,
                         fontsize=label_fontsize - 4.5)
    ax.set_ylim(0, top)
    ax.set_ylabel("Mean hours per month", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=GENIE_LIQUID_COLOR, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc=GENIE_ICE_COLOR, ec="none"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.35", lw=0.8,
                      hatch=ERA5_HATCH),
        plt.Rectangle((0, 0), 1, 1, fc="0.55", ec="none"),
        Line2D([0], [0], color="0.15", lw=1.5),
    ]
    ax.legend(handles, ["liquid containing", "ice only",
                        "ERA5 (left, stippled)", "ARM obs (right, solid)",
                        f"$\\pm$1 s.d. across {n_txt} ERA5 seasons"],
              fontsize=legend_fontsize, ncol=2, framealpha=0.9, loc="upper right")
    draw_threshold_box(ax, A, loc="upper left",
                       fontsize=legend_fontsize - 2.0)

    band_note = ""
    if show_residual:
        # ---- residual panel --------------------------------------------------
        d_liq = residual_values(e_liq, o_liq, residual_mode)
        d_ice = residual_values(e_ice, o_ice, residual_mode)
        # The monthly file is still digitised, so unlike the seasonal figures
        # this one does still have a band to draw.
        band_note = draw_residual_band(
            ax_r, residual_mode, OBS_MONTHLY_UNCERTAINTY_H,
            bars=((x - w / 2, o_liq), (x + w / 2, o_ice)), width=w, show=show_band)
        ax_r.bar(x - w / 2, d_liq, width=w, color=GENIE_LIQUID_COLOR,
                 edgecolor="white", linewidth=0.5)
        ax_r.bar(x + w / 2, d_ice, width=w, color=GENIE_ICE_COLOR,
                 edgecolor="white", linewidth=0.5)
        ax_r.axhline(0.0, color="0.3", lw=1.0)
        ax_r.set_ylabel(RESIDUAL_YLABEL[residual_mode], fontsize=label_fontsize)
        ax_r.tick_params(axis="both", labelsize=tick_fontsize)
        ax_r.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax_r.set_axisbelow(True)
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)
        tick_ax = ax_r
    else:
        tick_ax = ax
    tick_ax.set_xticks(x)
    tick_ax.set_xticklabels(
        [calendar.month_abbr[m] + ("\u2020" if n_per_month[k] < n_hi else "")
         for k, m in enumerate(shared)], fontsize=tick_fontsize)

    drop_note = ""
    if dropped_months:
        drop_txt = ", ".join(f"{calendar.month_abbr[m]} {y}"
                             for y, m in sorted(dropped_months))
        drop_note = (f"\u2020 season-months excluded from BOTH sides (ARM "
                     f"instrument problems): {drop_txt}")
    n_note = (f"mean over {n_hi} ERA5 seasons" if n_lo == n_hi
              else f"mean over {n_lo}\u2013{n_hi} ERA5 seasons per month")
    fig.suptitle(f"Monthly mean cloud hours, ERA5 against ARM observations "
                 f"\u2014 {series_label}\n{args.region}   |   {n_note}   |   "
                 f"all sky, precipitation included on both "
                 f"sides   |   percentage is the liquid-containing share of the "
                 f"month", fontsize=12, y=0.965)
    if show_residual:
        fig.subplots_adjust(top=0.90, bottom=0.115, left=0.09, right=0.985)
    else:
        fig.subplots_adjust(top=0.86, bottom=0.20, left=0.09, right=0.985)
    draw_figure_footnotes(fig, [drop_note, band_note])
    suffix = "" if show_residual else " - no-residual-panel"
    return _save_stack(fig, A, out_dir,
                       _residual_stem("monthly_era5_vs_obs", residual_mode), dpi,
                       suffix=suffix)


# ----------------------------------------------------------------------------
# Season/month-by-season residual tables -- the same numbers as the lower
# panel of fig_era5_vs_obs / fig_era5_vs_obs_allsky / fig_monthly_era5_vs_obs,
# typeset as a table rather than plotted, for pasting into a manuscript.
# ----------------------------------------------------------------------------

def _residual_header(mode: str) -> str:
    """Column-header fragment matching ``residual_values``' units."""
    return "% diff" if mode == "percent_diff" else "diff (h)"


def _residual_cell(v: float, mode: str) -> str:
    """One formatted table cell; an em dash where the obs denominator is 0."""
    if not np.isfinite(v):
        return "—"
    return f"{v:+,.0f} h" if mode == "hours" else f"{v:+.1f}%"


def _threshold_tag(A: Analysis) -> str:
    """``min_cloud_fraction`` and ``ice_fraction_min``, formatted for a file name.

    Requires ``phase_mode="fraction"`` -- the LWP-threshold mode has no single
    ``ice_fraction_min`` to report, and these tables are read against the
    fraction-mode comparison figures.
    """
    args = A.args
    pk = A.phase_kw
    if pk["mode"] != "fraction":
        raise ValueError(
            "percent-diff tables need phase_mode='fraction' so "
            "ice_fraction_min is defined for the file name")
    return f"mcf{args.min_cloud_fraction:g}_ifm{pk['ice_fraction_min']:g}"


def _render_residual_table(headers: list[str], rows: list[tuple],
                           avg_row: tuple | None = None,
                           font_size: float = 11):
    """A minimalist 'booktabs'-style table: horizontal rules only, no vertical
    lines, serif type -- the look of a typeset journal table rather than a
    spreadsheet grid.

    ``rows`` is a list of ``(label, [cell_strs...], flagged)`` tuples. A row
    with ``flagged=True`` gets an asterisk appended to its label and its
    values set in italics -- the table's equivalent of the diagonal-stripe
    hatch the comparison figures use for the same seasons. ``avg_row``, if
    given, is a ``(label, [cell_strs...])`` tuple drawn in bold below its own
    rule, and is left out of the flagging entirely (the caller excludes
    flagged rows from the mean before formatting it).

    Row height and margins scale with ``font_size`` (anchored at the
    long-standing default of 11 pt) so a larger or smaller font doesn't end
    up cramped or floating in oversized whitespace.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_cols = len(headers)
    n_data = len(rows)
    n_slots = 1 + n_data + (1 if avg_row is not None else 0)

    scale = font_size / 11.0
    row_h = 0.28 * scale    # inches
    top_m, bot_m = 0.14 * scale, 0.14 * scale
    fig_w = 5.4
    fig_h = top_m + bot_m + row_h * n_slots

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, fig_h)
    ax.axis("off")

    x_label = 0.04
    x_right = 0.97
    x_vals = (np.linspace(0.60, x_right, n_cols - 1) if n_cols > 2
             else np.array([x_right]))

    y0 = fig_h - top_m   # y of the top rule

    def rule(y, lw):
        ax.add_line(Line2D([0.02, 0.98], [y, y], color="black", linewidth=lw))

    rule(y0, 1.4)
    for slot in range(n_slots):
        yc = y0 - row_h * slot - row_h / 2.0
        is_header = slot == 0
        is_avg = avg_row is not None and slot == n_slots - 1
        if is_header:
            label, values, flagged, weight = headers[0], headers[1:], False, "normal"
        elif is_avg:
            label, values, flagged, weight = avg_row[0], avg_row[1], False, "bold"
        else:
            label, values, flagged = rows[slot - 1]
            weight = "normal"
        shown_label = f"{label}*" if flagged else label
        ax.text(x_label, yc, shown_label, ha="left", va="center",
               fontsize=font_size, family="serif", fontweight=weight)
        style = "italic" if flagged else "normal"
        for xv, val in zip(x_vals, values):
            ax.text(xv, yc, val, ha="right", va="center", fontsize=font_size,
                   family="serif", fontstyle=style, fontweight=weight)
        if is_header:
            rule(y0 - row_h, 0.8)
        if avg_row is not None and slot == n_slots - 2:
            rule(y0 - row_h * (slot + 1), 0.8)

    rule(y0 - row_h * n_slots, 1.4)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def _save_table(fig, out_dir, fname: str, dpi, pad_inches: float = 0.03):
    if out_dir is None:
        return fig
    path = Path(out_dir) / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches,
               facecolor="white")
    print(f"  -> {path}")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def table_era5_vs_obs_pct_diff(A: Analysis, obs_path=DEFAULT_OBS_FILE,
                               out_dir=None, surface_class: str = "arm_site",
                               residual_mode: str | None = None,
                               dpi: int = 350, font_size: float = 11):
    """Season-by-season residual table matching :func:`fig_era5_vs_obs`.

    One row per season, liquid-containing and ice-only side by side, in
    whichever units ``residual_mode`` (default from the run's
    ``--residual-mode``) gives :func:`residual_values` -- percent difference
    by default. Seasons flagged by the same rule the figure uses (missing
    hours over 5% of the season window) get an asterisk on the season label,
    italic values, and are excluded from the ``Average`` row.

    ``font_size`` controls the table's type size in points; row height and
    margins scale with it automatically.
    """
    args = A.args
    residual_mode = resolve_residual_mode(residual_mode, args)
    obs = load_observations(obs_path)
    labels, liquid, ice, _clear, season_h = season_phase_binary(A)
    code, _series_label = resolve_series_code(A.col, surface_class)

    era_years = [int(l.split("/")[0]) for l in labels]
    obs_liq_all, obs_ice_all = obs_binary(obs, exclude_precip=args.no_precip)
    obs_idx = {y: i for i, y in enumerate(obs["seasons"])}
    shared = [y for y in era_years if y in obs_idx]
    if not shared:
        raise ValueError("no season appears in both the ERA5 run and "
                         f"{obs_path}")
    ei = {y: i for i, y in enumerate(era_years)}
    e_liq = np.array([liquid[ei[y], code] for y in shared])
    e_ice = np.array([ice[ei[y], code] for y in shared])
    o_liq = np.array([obs_liq_all[obs_idx[y]] for y in shared])
    o_ice = np.array([obs_ice_all[obs_idx[y]] for y in shared])
    miss = np.array([obs["missing"][obs_idx[y]] for y in shared])
    s_h = np.array([_sh(season_h, ei[y]) for y in shared])
    missing_mask = miss > 0.05 * s_h

    d_liq = residual_values(e_liq, o_liq, residual_mode)
    d_ice = residual_values(e_ice, o_ice, residual_mode)

    hdr = _residual_header(residual_mode)
    rows = [(f"{y}/{(y + 1) % 100:02d}",
            [_residual_cell(d_liq[i], residual_mode),
             _residual_cell(d_ice[i], residual_mode)],
            bool(missing_mask[i]))
           for i, y in enumerate(shared)]
    keep = ~missing_mask
    avg_row = ("Average",
              [_residual_cell(np.nanmean(d_liq[keep]), residual_mode),
               _residual_cell(np.nanmean(d_ice[keep]), residual_mode)])

    fig = _render_residual_table(
        ["Season", f"Liquid {hdr}", f"Ice {hdr}"], rows, avg_row=avg_row,
        font_size=font_size)
    tag = "noprecip" if args.no_precip else "allsky"
    fname = (f"{args.region}_era5_vs_obs_{tag}_pct_diff_table_"
            f"{_threshold_tag(A)}.png")
    return _save_table(fig, out_dir, fname, dpi)


def table_era5_vs_obs_allsky_pct_diff(A: Analysis, obs_path=DEFAULT_OBS_FILE,
                                      out_dir=None, surface_class="arm_site",
                                      residual_mode: str | None = None,
                                      dpi: int = 350, font_size: float = 11):
    """Season-by-season residual table matching :func:`fig_era5_vs_obs_allsky`.

    Same layout as :func:`table_era5_vs_obs_pct_diff`; the observation side is
    the all-sky total (with-liquid + liquid-precip, ice-only + ice-precip),
    matching the unfiltered ERA5 run the figure requires.
    """
    args = A.args
    residual_mode = resolve_residual_mode(residual_mode, args)
    if args.no_precip:
        raise ValueError(
            "table_era5_vs_obs_allsky_pct_diff needs a run with "
            "no_precip=False, same as fig_era5_vs_obs_allsky.")

    obs = load_observations(obs_path)
    labels, liquid, ice, _clear, season_h = season_phase_binary(A)
    code, _series_label = resolve_series_code(A.col, surface_class)

    era_years = [int(l.split("/")[0]) for l in labels]
    obs_idx = {y: i for i, y in enumerate(obs["seasons"])}
    shared = [y for y in era_years if y in obs_idx]
    if not shared:
        raise ValueError(f"no season appears in both the run and {obs_path}")
    ei = {y: i for i, y in enumerate(era_years)}
    e_liq = np.array([liquid[ei[y], code] for y in shared])
    e_ice = np.array([ice[ei[y], code] for y in shared])
    o_liq_tot = np.array([obs["with_liquid"][obs_idx[y]]
                          + obs["liq_precip"][obs_idx[y]] for y in shared])
    o_ice_tot = np.array([obs["ice_only"][obs_idx[y]]
                          + obs["ice_precip"][obs_idx[y]] for y in shared])
    miss = np.array([obs["missing"][obs_idx[y]] for y in shared])
    s_h = np.array([_sh(season_h, ei[y]) for y in shared])
    missing_mask = miss > 0.05 * s_h

    d_liq = residual_values(e_liq, o_liq_tot, residual_mode)
    d_ice = residual_values(e_ice, o_ice_tot, residual_mode)

    hdr = _residual_header(residual_mode)
    rows = [(f"{y}/{(y + 1) % 100:02d}",
            [_residual_cell(d_liq[i], residual_mode),
             _residual_cell(d_ice[i], residual_mode)],
            bool(missing_mask[i]))
           for i, y in enumerate(shared)]
    keep = ~missing_mask
    avg_row = ("Average",
              [_residual_cell(np.nanmean(d_liq[keep]), residual_mode),
               _residual_cell(np.nanmean(d_ice[keep]), residual_mode)])

    fig = _render_residual_table(
        ["Season", f"Liquid {hdr}", f"Ice {hdr}"], rows, avg_row=avg_row,
        font_size=font_size)
    fname = (f"{args.region}_era5_vs_obs_allsky_split_pct_diff_table_"
            f"{_threshold_tag(A)}.png")
    return _save_table(fig, out_dir, fname, dpi)


def table_monthly_era5_vs_obs_pct_diff(A: Analysis,
                                       obs_path=DEFAULT_MONTHLY_OBS_FILE,
                                       out_dir=None,
                                       surface_class: str = "arm_site",
                                       residual_mode: str | None = None,
                                       exclude_months=GENIE_EXCLUDED_MONTHS,
                                       dpi: int = 220, font_size: float = 11):
    """Month-by-month residual table matching :func:`fig_monthly_era5_vs_obs`.

    One row per calendar month, liquid-containing and ice-only side by side.
    No seasons are dropped here -- the monthly file has no per-month
    "missing hours" figure the way the seasonal one does -- so there is no
    asterisk column and no average row, just the twelve (or fewer) months.
    """
    args = A.args
    residual_mode = resolve_residual_mode(residual_mode, args)
    if args.no_precip:
        raise ValueError(
            "table_monthly_era5_vs_obs_pct_diff needs a run with "
            "no_precip=False, same as fig_monthly_era5_vs_obs.")

    obs = load_monthly_observations(obs_path)
    months, liq, ice, _month_h = monthly_phase_binary(A, surface_class,
                                                      exclude_months)
    o_idx = {m: i for i, m in enumerate(obs["months"])}
    shared = [m for m in months if m in o_idx]
    if not shared:
        raise ValueError(f"no month appears in both the run and {obs_path}")
    mi = {m: i for i, m in enumerate(months)}
    e_liq = np.array([nanmean_quiet(liq[:, mi[m]]) for m in shared])
    e_ice = np.array([nanmean_quiet(ice[:, mi[m]]) for m in shared])
    o_liq = np.array([obs["with_liquid"][o_idx[m]] for m in shared])
    o_ice = np.array([obs["ice_only"][o_idx[m]] for m in shared])

    d_liq = residual_values(e_liq, o_liq, residual_mode)
    d_ice = residual_values(e_ice, o_ice, residual_mode)

    hdr = _residual_header(residual_mode)
    rows = [(calendar.month_abbr[m],
            [_residual_cell(d_liq[i], residual_mode),
             _residual_cell(d_ice[i], residual_mode)], False)
           for i, m in enumerate(shared)]

    fig = _render_residual_table(["Month", f"Liquid {hdr}", f"Ice {hdr}"], rows,
        font_size=font_size)
    fname = (f"{args.region}_monthly_era5_vs_obs_pct_diff_table_"
            f"{_threshold_tag(A)}.png")
    return _save_table(fig, out_dir, fname, dpi)


def fig_linear(A: Analysis, out_dir=None, dpi: int | None = None):
    """Linear LWP bins -- the physical-axis copy."""
    return figure(A, "linear", out_dir, dpi)


def fig_log(A: Analysis, out_dir=None, dpi: int | None = None):
    """Log-spaced LWP bins -- the copy that resolves the low-LWP end."""
    return figure(A, "log", out_dir, dpi)


ALL_FIGURES = (fig_linear, fig_log, fig_monthly_phase_fraction,
               fig_season_phase_stack, fig_season_phase_stack_site,
               fig_season_phase_binary)


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
    # Not gated on --bin-scale: that option selects between two BINNINGS of the
    # histogram, and this figure has no LWP axis to bin.
    for surface_class in args.monthly_class:
        fig_monthly_phase_fraction(A, out_dir=out_dir,
                                   surface_class=surface_class)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
