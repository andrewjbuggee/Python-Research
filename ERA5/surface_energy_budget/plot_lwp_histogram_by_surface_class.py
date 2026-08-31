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

REQUIRED_VARS = ("tcc", "tclw", "tciw", "siconc")

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


def month_window_hours(slots: list[tuple[int, int]]) -> np.ndarray:
    """Hours the season window contains in each of its calendar months.

    Counted from the day-slots actually in the window, not from the calendar, so
    a window that starts mid-month gives that month its true partial length
    rather than a full one. This is the denominator that turns a monthly
    occupancy fraction into hours per month.
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

    weights_2d = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])
    w_per_step = float(weights_2d.sum())

    n_unclassified = 0
    n_precip_removed = 0.0     # cloudy cell-hours the filter dropped
    n_cloudy_before = 0.0      # cloudy cell-hours before it
    site_class_counts = np.zeros(len(CLASS_ORDER) + 1, dtype=np.int64)

    read_vars = list(REQUIRED_VARS)
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

        tclw_g = block["tclw"].values[keep] * 1000.0        # kg m-2 -> g m-2
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
        "season_hours": season_hours,
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
NOTE_BOX = dict(boxstyle="round,pad=0.6", facecolor="#f5f5f2",
                edgecolor="#bfbfbf", linewidth=0.8)


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
        f"  ({col['season_hours']:,.0f} h per season)",
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


def draw_notes(ax_note, lines, title: str | None = None) -> None:
    """Render the side note.

    ``None`` entries are dropped, so a caller can build the list with optional
    rows inline. An empty STRING is kept and becomes a blank line -- that is how
    the note is grouped into blocks, so the two must not be conflated.
    """
    body = "\n".join(ln for ln in lines if ln is not None)
    text = f"{title}\n{body}" if title else body
    ax_note.text(0.0, 1.0, text, transform=ax_note.transAxes,
                 va="top", ha="left", fontsize=8.8, linespacing=1.6,
                 bbox=NOTE_BOX)


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
        pct_season = 100.0 * total_h / col["season_hours"]
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
    season_h = col["season_hours"]
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
        f"  ({col['season_hours']:,.0f} h per season)",
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

    missing = sorted(set(REQUIRED_VARS) - set(ds.data_vars))
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
              f"{sampled.min():,.0f} to {sampled.max():,.0f} of {season_h:,.0f}."
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


def _save_stack(fig, A, out_dir, stem, dpi):
    """Write a season-stack figure, matching the naming the other figures use."""
    if out_dir is None:
        return fig
    path = Path(out_dir) / f"{A.args.region}_{stem}_{A.tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def season_phase_hours(A: Analysis):
    """Cell-hours per phase, per season, for a TYPICAL cell of each class.

    Returns ``(season_labels, hours, cloudy, season_h)`` with ``hours`` shaped
    ``(n_season, n_class, n_phase)`` in ``SEASON_STACK_ORDER``.

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
    return labels, hours, cloudy, float(col["season_hours"])


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
        for xi, tot in zip(x, bottom):
            if not np.isfinite(tot) or tot <= 0:
                continue
            ax.text(xi, tot + 0.015 * top,
                    f"{tot:,.0f} h\n({100.0 * tot / season_h:.1f}%)",
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


def _stack_subtitle(A, season_h):
    pk = A.col["phase_kw"]
    return (f"cell-hours for a typical cell of each class   |   "
            f"season window {season_h:,.0f} h   |   "
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
                 f"{args.region}   |   season window {season_h:,.0f} h   |   "
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
    clear = np.clip(season_h - (liquid + ice), 0.0, None)
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
    for xi, cloud_h, bar_h in zip(x, tot_cloud, top):
        if not np.isfinite(cloud_h) or cloud_h <= 0:
            continue
        ax.text(xi, bar_h + 0.012 * headroom,
                f"{cloud_h:,.0f} h\n({100.0 * cloud_h / season_h:.1f}%)",
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
                 f"{season_h:,.0f} h   |   {note}   |   "
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
OBS_BAR_COLOR = "#d1495b"
ERA5_HATCH = "...."  # stippling that marks ERA5 bars apart from the solid obs bars

# Text sizes for fig_era5_vs_obs, exposed as function arguments so a notebook
# can bump them per call rather than editing the source.
DEFAULT_COMPARISON_LABEL_FONTSIZE = 13.0   # axis labels
DEFAULT_COMPARISON_TICK_FONTSIZE = 12.0    # tick labels on both axes
DEFAULT_COMPARISON_LEGEND_FONTSIZE = 11.5  # both panels' legends


def load_observations(path=DEFAULT_OBS_FILE, check: bool = True) -> dict:
    """Read the ARM seasonal hours table.

    Returns ``{"seasons": [...], <column>: array, ...}``. With ``check``, the
    per-season columns are averaged and compared against the record means the
    source figure's legend states; a departure of more than 100 h is reported,
    since the file may hold values digitised from a figure rather than the
    observations themselves.
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


def fig_era5_vs_obs(A: Analysis, obs_path=DEFAULT_OBS_FILE, out_dir=None,
                    dpi: int | None = None, surface_class: str = "arm_site",
                    label_fontsize: float = DEFAULT_COMPARISON_LABEL_FONTSIZE,
                    tick_fontsize: float = DEFAULT_COMPARISON_TICK_FONTSIZE,
                    legend_fontsize: float = DEFAULT_COMPARISON_LEGEND_FONTSIZE):
    """ERA5 beside the ARM observations, with the residual underneath.

    Upper panel: two stacked bars per season -- ERA5 on the left, observations
    on the right -- each split into liquid-containing and ice-only.

    Lower panel: ERA5 minus observations, one bar per category per season.

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
    from matplotlib.lines import Line2D

    args = A.args
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

    x = np.arange(len(shared))
    w = 0.38
    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(2.0 + 1.35 * len(shared), 9.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.10})

    # ERA5 bars carry the same liquid/ice colors as the obs bars but stippled
    # rather than solid, so the two datasets no longer need a second, easily
    # confused color (a colored outline) layered on top of the phase colors.
    for off, (lq, ic, name, stippled) in (
            (-w / 2, (e_liq, e_ice, "ERA5", True)),
            (+w / 2, (o_liq, o_ice, "ARM obs", False))):
        hatch = ERA5_HATCH if stippled else None
        ax.bar(x + off, lq, width=w,
               color="white" if stippled else GENIE_LIQUID_COLOR,
               edgecolor=GENIE_LIQUID_COLOR if stippled else "none",
               linewidth=0.7, hatch=hatch)
        ax.bar(x + off, ic, width=w, bottom=lq,
               color="white" if stippled else GENIE_ICE_COLOR,
               edgecolor=GENIE_ICE_COLOR if stippled else "none",
               linewidth=0.7, hatch=hatch)

    # Mark seasons where the observations are substantially incomplete: their
    # totals are not comparable however good the ERA5 side is.
    top = float(max((e_liq + e_ice).max(), (o_liq + o_ice).max())) * 1.20
    for xi, m in zip(x, miss):
        if m > 0.05 * season_h:
            ax.text(xi + w / 2, (o_liq + o_ice)[list(x).index(xi)] + 0.02 * top,
                    f"{m:,.0f} h\nmissing", ha="center", va="bottom",
                    fontsize=7.5, color=OBS_BAR_COLOR, linespacing=1.1)
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
    ]
    ax.legend(handles, ["liquid containing", "ice only",
                        "ERA5 (left, stippled)", "ARM obs (right, solid)"],
              fontsize=legend_fontsize, ncol=2, framealpha=0.9, loc="upper left")

    # ---- residual panel ----------------------------------------------------
    d_liq, d_ice = e_liq - o_liq, e_ice - o_ice
    ax_r.bar(x - w / 2, d_liq, width=w, color=GENIE_LIQUID_COLOR,
             edgecolor="white", linewidth=0.5, label="liquid containing")
    ax_r.bar(x + w / 2, d_ice, width=w, color=GENIE_ICE_COLOR,
             edgecolor="white", linewidth=0.5, label="ice only")
    ax_r.axhline(0.0, color="0.3", lw=1.0)
    # Digitisation uncertainty band, when the file still holds read-off values.
    ax_r.axhspan(-150, 150, color="0.85", zorder=0)
    ax_r.text(0.995, 0.04, "grey band: +/-150 h, the digitisation uncertainty "
              "of the observation file", transform=ax_r.transAxes, ha="right",
              va="bottom", fontsize=7.5, color="0.35")

    # Star the same seasons flagged "missing" in the upper panel, so the
    # residual there is not read as if it were on equal footing with the rest.
    missing_mask = miss > 0.05 * season_h
    if missing_mask.any():
        resid_span = float(max(np.abs(d_liq).max(), np.abs(d_ice).max(), 150.0))
        star_margin = 0.05 * resid_span
        for xi in x[missing_mask]:
            y_top = max(d_liq[xi], d_ice[xi], 0.0) + star_margin
            ax_r.text(xi, y_top, "*", ha="center", va="bottom",
                      fontsize=13, fontweight="bold", color="0.25")

    ax_r.set_ylabel("ERA5 - observations [h]", fontsize=label_fontsize)
    ax_r.tick_params(axis="both", labelsize=tick_fontsize)
    ax_r.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax_r.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_r.spines[sp].set_visible(False)
    r_handles, r_labels = ax_r.get_legend_handles_labels()
    if missing_mask.any():
        r_handles.append(Line2D([0], [0], marker="*", linestyle="None",
                                 markersize=11, color="0.25"))
        r_labels.append("season has ≥5% missing obs hours (top panel)")
    ax_r.legend(r_handles, r_labels, fontsize=legend_fontsize, ncol=1,
                framealpha=0.9, loc="upper right")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([f"{y}/{(y + 1) % 100:02d}" for y in shared],
                         rotation=45, ha="right", fontsize=tick_fontsize)

    pair = ("precipitating scenes EXCLUDED from both sides" if args.no_precip
            else "precipitation INCLUDED on both sides")
    note = f"   |   {dropped} ERA5 season(s) not in the obs file" if dropped else ""
    fig.suptitle(f"ERA5 against ARM observations \u2014 {series_label}\n"
                 f"{args.region}   |   {pair}   |   {precip_label(args)}{note}",
                 fontsize=12.5, y=0.965)
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.09, right=0.985)
    tag = "noprecip" if args.no_precip else "allsky"
    return _save_stack(fig, A, out_dir, f"era5_vs_obs_{tag}", dpi)


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
