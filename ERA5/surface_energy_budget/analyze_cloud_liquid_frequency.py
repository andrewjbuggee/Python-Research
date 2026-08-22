#!/usr/bin/env python3
"""How often is liquid water present in Arctic clouds?

Counts, over a season, how much of the time a cloudy scene contains cloud liquid
water, and reports it three ways:

    (1) OVERALL   totals and fraction over the whole requested record
    (2) MONTHLY   mean time and fraction per calendar month
    (3) DAILY     mean time and fraction per day of season

DEFINITIONS
===========
A grid cell at one hour is a **cloudy scene** when

    tcc >= --min-cloud-fraction        (default 1.0, i.e. completely overcast)
    AND (tclw > --lwp-threshold  OR  tciw > --iwp-threshold)

that is, the sky is covered and *some* condensate is present, liquid or ice.
Liquid is then counted as present when

    tclw > --lwp-threshold

ERA5's names for the two paths are ``tclw`` (total column cloud liquid water =
LWP) and ``tciw`` (total column cloud ice water = IWP), both stored NATIVELY in
kg m-2. ``--lwp-threshold`` and ``--iwp-threshold`` are given in g m-2 instead,
for readability against the g m-2 magnitudes quoted throughout this file
(1 g m-2 = 0.001 kg m-2); the conversion to kg m-2 happens internally, right
before each threshold is compared against the raw field.

WHAT THE REPORTED FRACTION IS, AND IS NOT
=========================================
It is (area-weighted cloudy cell-hours with liquid) / (area-weighted cloudy
cell-hours). That is a *conditional relative frequency*, which is the empirical
estimator of a conditional probability but is not one by definition. Three
caveats, in decreasing order of how much they matter:

  1. The unit is a CELL-HOUR, not a cloud. A 0.25 deg cell at one hour is one
     count, whether it holds one stratocumulus deck or a dozen cloud elements.
     "The fraction of clouds containing liquid" is therefore not what this
     computes; "the fraction of overcast cell-hours that contain liquid" is.
     The two differ whenever cloud size correlates with phase, which in the
     Arctic it does: mixed-phase decks are large and persistent.
  2. It is a census, not a sample. Every cell-hour in the window is counted, so
     there is no sampling error to put a confidence interval on. The spread
     shown in the figures is interannual variability across seasons, not
     uncertainty in the fraction.
  3. Cell-hours are strongly autocorrelated in space and time, so the effective
     number of independent scenes is far below the raw count. This does not bias
     the fraction, but it does mean the count is not a sample size.

Read as "of the overcast cell-hours in this window, X% contained cloud liquid",
the number is exact and needs no probabilistic reading at all.

TWO WAYS TO COUNT HOURS PER DAY, AND THEY ARE NOT THE SAME
==========================================================
Figures 2 and 5 both plot hours per day, from different sample units:

  fig 2  each season-day is collapsed to one AREA MEAN over the strip, and the
         median and IQR are taken across the 6 seasons. Answers "how cloudy was
         the strip, on average, on 1 August?"
  fig 5  the sample unit is one CELL-DAY -- one grid cell on one 1 August -- and
         the quantiles run over all cells and all seasons pooled. Answers "in a
         typical cell, how many of the 24 hours had a liquid-bearing cloud
         overhead?"

They diverge whenever the field is patchy: an area mean of 12 h is produced
equally by every cell being cloudy half the day and by half the strip being
cloudy all day. Fig 5 also has an IQR even for a single season, because its
spread is across cells.

TOTAL CONDENSATE FALLS IN WINTER; IT DOES NOT SHIFT PHASE AT FIXED MASS
=======================================================================
The natural reading of the condensate figure -- LWP collapses while IWP holds
steady, so liquid must be turning into ice -- is not what the data say. Barrow
strip, 6 seasons, cloudy scenes, medians across seasons (g m-2):

    month    LWP    IWP   snow   rain  total   TCWV(kg/m2)  t2m(C)
    Aug    134.7   32.8   63.2    5.2  235.9      14.72       1.3
    Nov     30.9   28.8   24.2    0.0   83.8       4.13     -13.1
    Feb      3.7   20.7   16.0    0.0   40.3       2.50     -24.7

Three things follow.

  1. TOTAL condensate falls by a factor of 5.9 from August to February. There is
     far less condensed water in a winter column, exactly as a naive reading of
     the figure suggests. Nothing is hiding.
  2. The control is VAPOUR, not phase. TCWV falls by a factor of 5.9 over the
     same months -- the same factor. Condensate is 1.6-2.0% of column vapour in
     every month of the season, so the entire seasonal cycle in condensate is
     the seasonal cycle in how much water the air can hold at all.
  3. The phase shift is real but FRACTIONAL. Frozen water (IWP + snow) goes from
     41% of total condensate in August to 91% in February. Its absolute amount
     still falls, by 62%. Liquid falls by 97%. So ice wins the competition for a
     shrinking pool of water; it does not gain mass at liquid's expense.

Why IWP alone looks so flat: ERA5's ``tciw`` is SUSPENDED cloud ice only. The
IFS carries prognostic rain and snow, so falling snow is bookkept in ``tcsw``,
and in this region snow water is comparable to or larger than cloud ice all
season. ``--include-precip`` puts tcsw and tcrw on the condensate figures; the
monthly table always prints them.

THE FOUR CONDENSATE SPECIES ARE DISJOINT, SO THEY MAY BE SUMMED
==============================================================
The ECMWF descriptions of tclw / tciw / tcrw / tcsw all carry the same
boilerplate, which lists "cloud water droplets, raindrops, ice crystals and
snow" as a four-way enumeration. They are four separate prognostic variables in
the IFS cloud scheme (clwc, ciwc, crwc, cswc), not nested categories, so rain is
NOT part of tclw and snow is NOT part of tciw.

That is checkable without trusting the prose: a subset can never exceed its
superset. Over the Barrow strip, 2020-2021, 29.4 M cell-hours:

    tcrw > tclw   in 0.020% of cell-hours, excess up to 48 g m-2
    tcsw > tciw   in 12.5% of cell-hours, excess up to 2522 g m-2

Both would be impossible under nesting, so both species are additive with the
cloud paths. (tcrw exceeds tclw rarely only because rain usually falls out of a
cloud that still holds liquid; the rare cases are rain below an evaporated
cloud base.)

tcsw is atmospheric, not the snowpack. Two independent checks: its magnitude
here is 6-35 g m-2, whereas an Arctic snowpack is 50,000-150,000 g m-2 water
equivalent; and it peaks in late summer and falls through winter, the opposite
of an accumulating pack. Surface snow is a different variable (snow depth, sd).

TCC IS INSTANTANEOUS, SO "HOURS" MEANS "HOURLY SAMPLES"
=======================================================
tcc, tclw, tciw and the rest are instantaneous fields valid at the top of each
hour, unlike the ms* fluxes which are means over the preceding hour. Counting
them gives 24 samples per cell-day, and "10 hours" is really "10 of 24 samples
met the condition".

That is an unbiased estimate of the FRACTION OF THE DAY spent in the state,
because the sampling grid is fixed and independent of the cloud field. What it
cannot do is measure duration or contiguity: 10 samples could be one 10-hour
deck or ten isolated hours.

How badly the hourly grid aliases the real thing is measurable, by extracting
the actual episodes -- runs of consecutive samples in one cell, discarding any
that touch a gap in the record. Barrow, 7 seasons, tcc = 1 (figure 7):

    mean duration                       6.58 h
    median duration                     3 h
    episodes lasting exactly 1 sample   34.1%
    share of total overcast TIME in those 1-hour episodes   5.2%

The mean alone flatters the sampling. The median is 3 h, and a third of all
episodes are a single sample -- which is exactly the signature of structure at
or below the sampling scale. A memoryless process with the same measured
persistence would produce 15.1% singletons, so there is roughly twice the
short-episode population that hour-to-hour persistence alone implies.

What rescues the hour count is that those singletons carry only 5.2% of the
total overcast time. Occupancy -- the fraction of the day overcast, which is all
these figures claim -- is exposed to that 5%, not to the 34%.

MUCH OF THE SPIKE IS THE THRESHOLD, NOT THE CLOUD. tcc = 1 exactly is a knife
edge, and about a tenth of all cell-hours sit in [0.99, 1). A cell drifting
across that line manufactures 1-hour episodes out of a deck that never broke.
Relaxing the threshold separates the two (``--episode-scan`` prints this):

    tcc >=   base rate   mean h   median h   1-h eps   memoryless   time in 1-h
      1.00       0.519     6.58          3     34.1%        15.1%          5.2%
      0.99       0.626     9.84          4     27.5%        10.1%          2.8%
      0.95       0.750    18.29          7     19.6%         5.4%          1.1%
      0.90       0.803    23.92          8     17.1%         4.1%          0.7%
      0.50       0.924    57.86         16     13.2%         1.5%          0.2%

Mean duration rises from 6.6 h to 23.9 h between tcc = 1 and tcc >= 0.9, and the
time held in 1-hour episodes falls from 5.2% to 0.7%. The excess over the
memoryless benchmark never vanishes, so some genuine short-timescale structure
is there at every threshold -- but the strict criterion is responsible for most
of what looks like flicker.

THE THRESHOLD MATTERS, AND HOW MUCH DEPENDS ON THE SEASON
=========================================================
A zero threshold is not as uninformative as it first appears, but it is
seasonally uneven. Barrow strip, six seasons 2019/20-2024/25 (Aug-Mar), median
across seasons:

    month     cloudy cell-hours with liquid, at LWP > 0
    Aug-Sep    99.9%     saturated: says nothing beyond "it is not winter"
    Oct        98.8%
    Nov        96.5%
    Dec        82.3%
    Jan        66.9%
    Feb        62.0%     genuinely discriminating
    Mar        74.6%

In late summer nearly every overcast scene carries at least trace liquid, so a
zero threshold measures ERA5's numerical floor. By midwinter, liquid is absent
from a third of overcast scenes, and the same threshold becomes a real
measurement. The whole-record figure at LWP > 0 is 86.7%.

Raising the threshold lowers the fraction everywhere, and by a lot (same six
seasons, whole record):

    LWP >  0 g m-2   86.7% of cloudy cell-hours
    LWP >  1 g m-2   70.1%
    LWP >  5 g m-2   60.9%
    LWP > 10 g m-2   55.4%
    LWP > 20 g m-2   48.3%
    LWP > 50 g m-2   35.3%

If the aim is comparison against ground-based retrievals, pick a threshold near
their detection limit: microwave radiometer LWP uncertainty is roughly
10-25 g m-2, so ``--lwp-threshold 10`` to ``25``. Every run prints the
sensitivity table above for its own data, so the dependence is never hidden.

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
--season-end MM-DD           Last day, inclusive (default 03-31). The window MAY
                             wrap the new year, which this default does.
--years SPEC                 '2019', '2019-2025', or '2000,2019-2020'. For a
                             wrapping window this is the year the season STARTS
                             in. Default: every year present.

Thresholds
----------
--lwp-threshold G            Liquid water path above which liquid counts as
                             present, g m-2 (default 0.0; see the warning).
--iwp-threshold G            Ice water path used in the cloudy test, g m-2
                             (default 0.0).
--min-cloud-fraction F       Total cloud cover at or above which a scene counts
                             as cloudy (default 1.0, fully overcast).

Domain
------
--ocean-only                 Drop land cells. OFF by default: cloud properties
                             are defined over land too, and nothing here is
                             specific to the surface type.

Output
------
--output-dir PATH            Where figures go (default ./figures).
--dpi N                      Figure resolution (default 350).
--show                       Open interactive windows.
--no-figures                 Print the tables only.

Examples
--------
    python analyze_cloud_liquid_frequency.py --region barrow --years 2019

    python analyze_cloud_liquid_frequency.py --region barrow \
        --years 2019-2020 --lwp-threshold 10
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from seb_analysis_common import (
    add_data_source_args,
    area_weights,
    load_seb_data,
    resolve_region_dir,
    season_calendar,
    season_slot_of,
    season_year_of,
    weighted_quantiles,
)

REF_YEAR_START = 2000
REF_YEAR_WRAP = (1999, 2000)

# Thresholds used for the automatic sensitivity table, in g m-2.
SENSITIVITY_THRESHOLDS = (0.0, 1.0, 5.0, 10.0, 20.0, 50.0)

HOURS_PER_STEP = 1.0  # ERA5 hourly data: one time step is one hour

# Longest episode tracked exactly; anything longer is lumped into the top
# slot. Multi-day overcast does occur here, so this is deliberately roomy.
MAX_RUN_H = 500

# Cloud-cover thresholds for --episode-scan.
EPISODE_SCAN_THRESHOLDS = (1.0, 0.99, 0.95, 0.90, 0.70, 0.50)


# Darker grey for "cloudy" so it separates cleanly from the liquid series and
# from the white background.
CLOUD_COLOR = "#5C6B7A"
LIQUID_COLOR = "#1B7FBD"
ICE_COLOR = "#8C6BB1"
SNOW_COLOR = "#BCA6D6"   # falling snow: same family as cloud ice, lighter
RAIN_COLOR = "#7FC4E0"   # falling rain: same family as cloud liquid, lighter


def parse_month_day(text: str) -> tuple[int, int]:
    from datetime import date
    try:
        mm, dd = text.split("-")
        month, day = int(mm), int(dd)
        date(2000, month, day)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid MM-DD") from None
    return month, day


def parse_years(text: str) -> tuple[int, ...]:
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


def parse_layout(text: str) -> tuple[int, int]:
    """'2x4' or '2,4' -> (2, 4)."""
    parts = str(text).replace(",", "x").split("x")
    try:
        r, c = (int(p) for p in parts)
        if r < 1 or c < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a layout like 2x4") from None
    return r, c


def build_masks(ds, lwp_thr, iwp_thr, min_cf, ocean_only):
    """Boolean (time, lat, lon) masks over the whole record.

    ``lwp_thr`` and ``iwp_thr`` are in g m-2; ``tclw``/``tciw`` are ERA5's
    native kg m-2, so both are converted here before comparison.

    Returns ``(cloudy, liquid, liquid_any, valid)``:
      cloudy      overcast AND some condensate of either phase
      liquid      cloudy AND liquid above threshold  (conditional)
      liquid_any  liquid above threshold, REGARDLESS of cloud cover
      valid       every cell considered at all
    ``liquid_any`` is what an unconditional "how often is liquid present"
    question needs; ``liquid`` is the conditional one.
    """
    tcc = ds["tcc"].values
    tclw_g = ds["tclw"].values * 1000.0
    tciw_g = ds["tciw"].values * 1000.0

    valid = np.isfinite(tcc) & np.isfinite(tclw_g) & np.isfinite(tciw_g)
    if ocean_only:
        valid &= np.isfinite(ds["siconc"].values)

    cloudy = valid & (tcc >= min_cf) & ((tclw_g > lwp_thr) | (tciw_g > iwp_thr))
    liquid = cloudy & (tclw_g > lwp_thr)
    liquid_any = valid & (tclw_g > lwp_thr)
    return cloudy, liquid, liquid_any, valid


def season_index(ds, season, years):
    """Time indices inside the season window, plus their slot and season year."""
    times = ds["valid_time"].values
    slots = season_calendar(*season)
    slot = season_slot_of(times, slots)
    syear = season_year_of(times, season[0])
    keep = slot >= 0
    if years is not None:
        keep &= np.isin(syear, list(years))
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        raise ValueError("No time steps inside the requested season and years.")
    return idx, slot, syear, slots


def season_coverage(times, idx, slot, syear, n_slots):
    """Fraction of the season's day-slots each season actually contains."""
    cov = {}
    for y in sorted(set(syear[idx].tolist())):
        sel = idx[syear[idx] == y]
        cov[y] = len(np.unique(slot[sel])) / n_slots
    return cov


def per_season_daily(ds, idx, slot, syear, seasons, n_slots,
                     cloudy, liquid, liquid_any, valid, w_yx):
    """Reduce to arrays indexed [season, day-slot].

    Keeping the season axis intact is what makes a median and an interquartile
    band across years possible; collapsing to a single mean here would throw away
    exactly the spread the figures are meant to show. Cells are cos(latitude)
    weighted throughout, so a value is an area mean, not a cell count.

    UNITS. Every condensate path returned here is g m-2, converted from ERA5's
    native kg m-2 at source so that no caller has to remember to scale it. TCWV
    stays kg m-2 and t2m stays kelvin.
    """
    n_s, n_d = len(seasons), n_slots
    shape = (n_s, n_d)
    # tcsw/tcrw are the PRECIPITATING condensate. The IFS carries prognostic
    # rain and snow, so tciw holds suspended cloud ice only and falling snow is
    # bookkept separately -- which matters a lot in an Arctic winter, when most
    # of the frozen water in the column is snow, not cloud ice. tcwv and t2m are
    # carried so the seasonal condensate cycle can be read against the vapour
    # available to make it.
    extras = {"tcsw": "snow_cloudy", "tcrw": "rain_cloudy",
              "tcwv": "tcwv_cloudy", "t2m": "t2m_cloudy"}
    extras = {v: k for v, k in extras.items() if v in ds}
    # tcsw/tcrw are condensate paths in ERA5's kg m-2, like tclw/tciw, so they
    # are reported in g m-2 with everything else. tcwv stays kg m-2 (a column
    # vapour path of 15 kg m-2 is the conventional unit) and t2m stays kelvin.
    extra_scale = {"snow_cloudy": 1000.0, "rain_cloudy": 1000.0,
                   "tcwv_cloudy": 1.0, "t2m_cloudy": 1.0}
    out = {k: np.full(shape, np.nan) for k in
           ("cloudy_h", "liquid_h", "liquid_any_h", "frac",
            "lwp_cloudy", "iwp_cloudy", *extras.values())}
    s_pos = {y: i for i, y in enumerate(seasons)}

    tclw = ds["tclw"].values
    tciw = ds["tciw"].values
    extra_a = {k: ds[v].values for v, k in extras.items()}

    for y in seasons:
        for d in np.unique(slot[idx[syear[idx] == y]]):
            sel = idx[(syear[idx] == y) & (slot[idx] == d)]
            if sel.size == 0:
                continue
            i_s, i_d = s_pos[y], int(d)
            w = np.broadcast_to(w_yx, cloudy[sel].shape)
            w_valid = float((valid[sel] * w).sum())
            if w_valid <= 0:
                continue
            n_steps = sel.size  # hours represented by this season-day

            c_w = float((cloudy[sel] * w).sum())
            l_w = float((liquid[sel] * w).sum())
            la_w = float((liquid_any[sel] * w).sum())

            # Area-weighted mean hours per cell for this day: the weighted count
            # divided by the weight of one full day of valid cells.
            per_cell_day = w_valid / n_steps
            out["cloudy_h"][i_s, i_d] = c_w / per_cell_day
            out["liquid_h"][i_s, i_d] = l_w / per_cell_day
            out["liquid_any_h"][i_s, i_d] = la_w / per_cell_day
            out["frac"][i_s, i_d] = (l_w / c_w) if c_w > 0 else np.nan

            # Mean condensate WHEN CLOUDY, area weighted.
            cm = cloudy[sel]
            if cm.any():
                # Scale AFTER the weighted sum rather than converting the whole
                # array, which would double the memory for no benefit.
                out["lwp_cloudy"][i_s, i_d] = (
                    float((tclw[sel] * cm * w).sum()) / c_w * 1000.0)
                out["iwp_cloudy"][i_s, i_d] = (
                    float((tciw[sel] * cm * w).sum()) / c_w * 1000.0)
                for k, a in extra_a.items():
                    out[k][i_s, i_d] = (float((a[sel] * cm * w).sum()) / c_w
                                        * extra_scale[k])
    return out


def contiguous_segments(times, idx):
    """Split ``idx`` into blocks whose timestamps step by exactly one hour.

    Season boundaries and any gap in the archive break a block. Runs must not be
    measured across a break, or two separate episodes either side of a missing
    week get glued into one implausibly long one.
    """
    t = times[idx].astype("datetime64[h]").astype("int64")
    brk = np.nonzero(np.diff(t) != 1)[0] + 1
    return [s for s in np.split(idx, brk) if s.size >= 3]


def episode_statistics(times, idx, masks, w_yx, month_of):
    """Length distribution of consecutive-True episodes, per cell.

    Returns ``{name: {"hist": H, "censored": c, "total": n}}`` where ``H`` is a
    cos(latitude) weighted count indexed [month, length_in_hours], length 0
    unused. Also returns hourly persistence per month.

    CENSORING. An episode already under way at the start of a contiguous block,
    or still under way at its end, has an unknown true length and is discarded
    rather than counted short. The discarded fraction is reported so the size of
    that correction is visible.

    WHAT THIS CANNOT SEE. The sampling grid is hourly, so an episode shorter
    than one hour is either missed entirely or recorded as one sample. The
    distribution is therefore censored from below at 1 h, and a large spike at
    exactly 1 h would be the signature of unresolved flicker. That the spike is
    small is evidence against flicker, not proof of its absence.
    """
    w_flat = np.asarray(w_yx).ravel()
    n_month = 13
    out = {}
    for name in masks:
        out[name] = {"hist": np.zeros((n_month, MAX_RUN_H + 1)),
                     "censored": 0.0, "total": 0.0,
                     "pair_on": np.zeros(n_month), "pair_on_on": np.zeros(n_month)}

    for seg in contiguous_segments(times, idx):
        mon = month_of[seg]
        for name, mask in masks.items():
            m = mask[seg].reshape(seg.size, -1)
            T = seg.size
            z = np.zeros((1, m.shape[1]), np.int8)
            d = np.diff(np.vstack([z, m.astype(np.int8), z]), axis=0)
            si = np.argwhere(d == 1)
            ei = np.argwhere(d == -1)
            if si.size == 0:
                continue
            si = si[np.lexsort((si[:, 0], si[:, 1]))]
            ei = ei[np.lexsort((ei[:, 0], ei[:, 1]))]
            keep = (si[:, 0] > 0) & (ei[:, 0] < T)
            rec = out[name]
            rec["total"] += si.shape[0]
            rec["censored"] += int((~keep).sum())
            if keep.any():
                L = np.minimum(ei[keep, 0] - si[keep, 0], MAX_RUN_H)
                cell = si[keep, 1]
                np.add.at(rec["hist"], (mon[si[keep, 0]], L), w_flat[cell])

            # Hourly persistence, attributed to the month of the earlier sample.
            prev, cur = m[:-1], m[1:]
            wm = np.broadcast_to(w_flat, prev.shape)
            np.add.at(rec["pair_on"], mon[:-1], (prev * wm).sum(axis=1))
            np.add.at(rec["pair_on_on"], mon[:-1], ((prev & cur) * wm).sum(axis=1))
    return out


def pd_month(times):
    """Calendar month of each time step, as a plain int array."""
    return times.astype("datetime64[M]").astype(int) % 12 + 1


def run_summary(hist_1d):
    """Weighted mean, median and 1-hour share from a length histogram."""
    L = np.arange(hist_1d.size)
    n = hist_1d.sum()
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    mean = float((L * hist_1d).sum() / n)
    c = np.cumsum(hist_1d) / n
    median = float(np.searchsorted(c, 0.5))
    return mean, median, float(hist_1d[1] / n)


def pooled_cell_day_hours(idx, slot, syear, seasons, n_slots, masks, valid,
                          w_yx, qs=(0.25, 0.50, 0.75)):
    """Quantiles of hours-per-day over the pooled CELL x SEASON sample.

    Different statistic from ``per_season_daily``. There, each season-day is
    first collapsed to one area-mean number and the spread is across seasons.
    Here the sample unit is a single (grid cell, season) pair -- one cell-day --
    and the quantiles are taken over all of them at once. So for day slot
    "Aug 01" the sample is every cell in the strip on every 1 August in the
    record, and the median answers "in a typical cell-day, how many of the 24
    hours had the event?".

    The two differ whenever the field is spatially patchy: an area mean of 12 h
    is produced equally by every cell being cloudy half the day and by half the
    strip being cloudy all day. The median over cell-days separates those.

    Quantiles are cos(latitude) weighted, so a 70N cell counts for about twice a
    80N one. Cells are required to be valid for the whole day; a season-day with
    fewer than 24 stored hours is scaled to a 24-hour equivalent.
    """
    groups: dict[tuple[int, int], list[int]] = {}
    for i in idx:
        groups.setdefault((int(syear[i]), int(slot[i])), []).append(i)

    w_flat = w_yx.ravel()
    out = {k: np.full((n_slots, len(qs)), np.nan) for k in masks}
    for d in range(n_slots):
        members = [(y, groups[(y, d)]) for y in seasons if (y, d) in groups]
        if not members:
            continue
        for name, mask in masks.items():
            vals, wts = [], []
            for _, sel in members:
                sel = np.asarray(sel)
                ok = valid[sel].all(axis=0).ravel()     # cell good all day
                if not ok.any():
                    continue
                hours = mask[sel].sum(axis=0).ravel() * (24.0 / sel.size)
                vals.append(hours[ok])
                wts.append(w_flat[ok])
            if not vals:
                continue
            out[name][d] = weighted_quantiles(
                np.concatenate(vals), np.concatenate(wts), qs)
    return out


def monthly_median_lwp(ds, idx, month_of, cloudy, w_yx, months, lwp_thr):
    """cos(latitude) weighted median LWP over LIQUID cloudy cell-hours, per month.

    Restricted to cell-hours where LWP exceeds ``lwp_thr`` -- the same
    "liquid" test used everywhere else in this script -- so an overcast
    ice-only scene (LWP at or below threshold) no longer drags the median
    toward zero. That ice-only population is already reported separately, as
    the histogram's leftmost bar; folding it back into this median would
    double-count the same information the wrong way.

    Pooled over seasons rather than averaged season by season: a median is not
    additive, so pooling is the honest way to get one number per month.
    """
    tclw_g = ds["tclw"].values * 1000.0
    thr_g = lwp_thr    # already g m-2
    w_flat = np.asarray(w_yx).ravel()
    out = {}
    for m in months:
        sel = idx[month_of[idx] == m]
        if sel.size == 0:
            out[m] = float("nan")
            continue
        cm = cloudy[sel].reshape(sel.size, -1)
        x_all = tclw_g[sel].reshape(sel.size, -1)
        liquid = cm & (x_all > thr_g)
        if not liquid.any():
            out[m] = float("nan")
            continue
        x = x_all[liquid]
        w = np.broadcast_to(w_flat, liquid.shape)[liquid]
        out[m] = weighted_quantiles(x, w, (0.5,))[0]
    return out


def lwp_to_bar_x(v, edges):
    """Where an LWP value falls in the histogram's categorical bar coordinate.

    Bar 0 is the at-or-below-threshold bar, bar 1 the underflow, bars 2..len(e)
    the log-spaced interior, bar len(e)+1 the overflow. Inside the interior the
    position is interpolated on the log axis, so the line lands where the eye
    expects it rather than snapping to a bar centre.
    """
    if not np.isfinite(v):
        return None
    if v < edges[0]:
        return 1.0
    if v >= edges[-1]:
        return float(len(edges) + 1)
    i = int(np.digitize(v, edges))            # edges[i-1] <= v < edges[i]
    span = np.log10(edges[i]) - np.log10(edges[i - 1])
    f = (np.log10(v) - np.log10(edges[i - 1])) / span
    return (i + 1) - 0.5 + f


def lwp_bin_edges(lo_g, hi_g, n_bins):
    """Log-spaced interior edges, in g m-2.

    Log rather than linear because the distribution spans four decades and is
    piled up at the bottom: in cloudy scenes the median LWP is 107 g m-2 in
    August and 0.12 g m-2 in January. Linear bins wide enough for August put
    every winter scene in the first bar, which is the one thing the figure is
    supposed to resolve. Bars below ``lo_g`` and above ``hi_g`` are added by the
    histogram itself, so nothing is dropped.
    """
    return np.geomspace(lo_g, hi_g, n_bins + 1)


def monthly_lwp_histogram(ds, idx, slot, syear, seasons, slots, cloudy, valid,
                          w_yx, edges_g, days_in_month, lwp_thr):
    """Hours per cell per month, binned by LWP. Indexed [season, month, bin].

    Bin layout, length ``len(edges_g) + 2``:

        0                    no liquid at all: an overcast scene of ice alone
        1                    liquid present but below the first edge
        2 .. len(edges_g)    the supplied interior bins
        len(edges_g) + 1     above the top edge

    The ice-only population is split out rather than merged into the first bin
    because it is a different kind of scene, not merely a small-LWP one, and in
    midwinter it is a third of all overcast hours.

    Hours are normalised the same way as the monthly bars: a per-day rate scaled
    to the calendar length of the month, so a season that sampled only part of a
    month is not counted as a short month.
    """
    mon = slot_months(slots)
    months = list(dict.fromkeys(mon.tolist()))
    n_bin = len(edges_g) + 2                      # none + under + interior + over
    H = np.full((len(seasons), len(months), n_bin), np.nan)

    tclw_g = ds["tclw"].values * 1000.0
    w_flat = w_yx.ravel()
    s_month = {}                                  # (season, month) -> time idx
    for i in idx:
        s_month.setdefault((int(syear[i]), int(mon[slot[i]])), []).append(i)
    days_seen: dict[tuple[int, int], set] = {}
    for i in idx:
        days_seen.setdefault((int(syear[i]), int(mon[slot[i]])), set()).add(int(slot[i]))

    for si, y in enumerate(seasons):
        for mi, m in enumerate(months):
            sel = s_month.get((y, m))
            if not sel:
                continue
            sel = np.asarray(sel)
            cm = cloudy[sel]
            vm = valid[sel]
            w = np.broadcast_to(w_flat, (sel.size, w_flat.size))
            # Weight of one full time step of valid cells: the denominator that
            # turns a weighted cell-hour count into hours per cell.
            per_step = float((vm.reshape(sel.size, -1) * w).sum()) / sel.size
            if per_step <= 0:
                continue
            x = tclw_g[sel].reshape(sel.size, -1)[cm.reshape(sel.size, -1)]
            ww = w[cm.reshape(sel.size, -1)]
            counts = np.zeros(n_bin)
            b = np.digitize(x, edges_g) + 1       # 1 = under, len(edges)+1 = over
            b[x <= lwp_thr] = 0                   # ice-only scenes; lwp_thr already g m-2
            np.add.at(counts, b, ww)
            n_days = len(days_seen[(y, m)])
            H[si, mi] = counts / per_step * (days_in_month[m] / n_days)
    return months, H


def slot_months(slots) -> np.ndarray:
    return np.array([m for m, _ in slots])


def monthly_from_daily(daily, slots, days_in_month):
    """Aggregate the [season, day] arrays to [season, month].

    Hours per month are formed as a per-DAY rate times the calendar length of
    the month, never as a raw sum divided by the number of seasons. A season
    that covers only part of a month would otherwise drag that month's mean
    down in proportion to how little of it was sampled -- which is a sampling
    artefact, not a property of the atmosphere.
    """
    mon = slot_months(slots)
    months = list(dict.fromkeys(mon.tolist()))
    n_s = next(iter(daily.values())).shape[0]
    out = {k: np.full((n_s, len(months)), np.nan) for k in daily}
    for j, m in enumerate(months):
        cols = np.nonzero(mon == m)[0]
        for k, arr in daily.items():
            block = arr[:, cols]
            with np.errstate(invalid="ignore"):
                rate = np.nanmean(block, axis=1)  # per-day mean over days present
            if k.endswith("_h"):
                rate = rate * days_in_month[m]    # scale to a full month
            out[k][:, j] = rate
    return months, out


def med_iqr(a):
    """Median and 25th/75th percentiles down the season axis."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        return (np.nanpercentile(a, 50, axis=0),
                np.nanpercentile(a, 25, axis=0),
                np.nanpercentile(a, 75, axis=0))


def nanmean_quiet(a, axis=None):
    """np.nanmean without the "Mean of empty slice" RuntimeWarning.

    An all-NaN slice is expected here, not a mistake: a season that never
    covered a given month leaves that month empty, and it should stay NaN
    silently. np.errstate does not help -- numpy raises this through the
    warnings module, not the floating-point error state.
    """
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _cloud_criterion_label(min_cf: float) -> str:
    """Say '= 1' when the threshold is the maximum, '>= x' otherwise.

    Total cloud cover is bounded at 1, so ">= 1" is the same set as "= 1" and
    reads as though partial values could qualify.
    """
    if min_cf >= 1.0:
        return "total cloud cover = 1"
    return f"total cloud cover $\\geq$ {min_cf:g}"


def _season_span(seasons) -> str:
    """'2019/20 - 2024/25' rather than '2019-2025'.

    A season labelled 2024 ends in 2025, so writing the range by start years
    alone implies one more season than exists.
    """
    if len(seasons) == 1:
        y = seasons[0]
        return f"season {y}/{(y + 1) % 100:02d}"
    a, b = seasons[0], seasons[-1]
    return (f"{len(seasons)} seasons: {a}/{(a + 1) % 100:02d}"
            f"\u2013{b}/{(b + 1) % 100:02d}")


class Analysis(SimpleNamespace):
    """Everything the figures need, computed once.

    Built by :func:`prepare`. Holding it in one object is what lets a
    notebook load the archive in one cell and then redraw any single figure
    without repeating the load, which takes minutes.
    """


def _emit(fig, A, out_dir, stem, dpi=None):
    """Save a figure if asked, and hand it back either way."""
    if out_dir is None:
        return fig
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{A.args.region}_{stem}.png"
    fig.savefig(path, dpi=dpi or A.args.dpi, bbox_inches="tight")
    print(f"  -> {path}")
    return fig


def _finish(ax, ylab, ylim=None):
    ax.set_ylabel(ylab, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _daily_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def prepare(argv=None, args=None, **overrides):
    """Load the archive and compute everything the figures share.

    ``argv`` takes the same strings as the command line; ``overrides`` sets
    individual options by name, e.g. ``prepare(region="barrow",
    years=(2019, 2020))``. Returns an :class:`Analysis`.

    This is the slow step -- it reads every file for the region and holds the
    subset in memory -- so a notebook should call it once and keep the result.
    """
    if args is None:
        args = parse_args([] if argv is None else argv)
    for k, v in overrides.items():
        if not hasattr(args, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(args, k, v)

    print("=" * 72)
    print("Cloud liquid water frequency")
    print("=" * 72)

    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)

    needed = ["tcc", "tclw", "tciw"] + (["siconc"] if args.ocean_only else [])
    missing = [v for v in needed if v not in ds]
    if missing:
        raise KeyError(f"dataset is missing {missing}")

    (m0, d0), (m1, d1) = args.season_start, args.season_end
    wraps = (m1, d1) < (m0, d0)
    print(f"  Source     : {region_dir}")
    print(f"  Season     : {m0:02d}-{d0:02d} to {m1:02d}-{d1:02d}"
          + ("  (wraps the new year)" if wraps else ""))
    print(f"  Cloudy scene: tcc >= {args.min_cloud_fraction:g} AND "
          f"(LWP > {args.lwp_threshold:g} or IWP > {args.iwp_threshold:g}) g m-2")
    print(f"  Liquid      : LWP > {args.lwp_threshold:g} g m-2 "
          f"({args.lwp_threshold / 1000:g} kg m-2)")
    print(f"  Domain      : {'ocean only' if args.ocean_only else 'all cells, land included'}")

    # Optional diagnostics: the precipitating condensate and the vapour and
    # temperature that set how much condensate there can be. Absent from older
    # downloads, so they are added only when present rather than being required.
    optional = [v for v in ("tcsw", "tcrw", "tcwv", "t2m") if v in ds]
    ds = ds[needed + optional].compute()

    idx, slot, syear, slots = season_index(
        ds, (args.season_start, args.season_end), args.years)

    cov = season_coverage(ds["valid_time"].values, idx, slot, syear, len(slots))
    print(f"\n  Seasons found ({len(cov)}), coverage of the "
          f"{len(slots)}-day window:")
    for y, f in cov.items():
        flag = "" if f >= args.min_season_coverage else "   (excluded)"
        print(f"    {y}/{(y+1) % 100:02d}: {100*f:5.1f}%{flag}")
    seasons = [y for y, f in cov.items() if f >= args.min_season_coverage]
    if not seasons:
        raise ValueError("no season meets --min-season-coverage "
                         f"{args.min_season_coverage}")
    idx = idx[np.isin(syear[idx], seasons)]
    print(f"  Using {len(seasons)} season(s): "
          f"{', '.join(f'{y}/{(y+1)%100:02d}' for y in seasons)}")

    cloudy, liquid, liquid_any, valid = build_masks(
        ds, args.lwp_threshold, args.iwp_threshold,
        args.min_cloud_fraction, args.ocean_only)

    lat = ds["latitude"].values
    w_yx = np.broadcast_to(area_weights(ds).values[:, None],
                           (lat.size, ds["longitude"].size))
    n_cells = int(valid[idx].any(axis=0).sum())
    print(f"  Cells       : {n_cells:,}   time steps: {idx.size:,}")

    daily = per_season_daily(ds, idx, slot, syear, seasons, len(slots),
                             cloudy, liquid, liquid_any, valid, w_yx)
    dim = {m: calendar.monthrange(2001 if m != 2 else 2000, m)[1]
           for m in range(1, 13)}
    months, monthly = monthly_from_daily(daily, slots, dim)


    pooled = pooled_cell_day_hours(
        idx, slot, syear, seasons, len(slots),
        {"cloud_liquid": liquid, "liquid_any": liquid_any, "cloudy": cloudy},
        valid, w_yx)
    med = {k: med_iqr(v)[0] for k, v in monthly.items()}
    month_of_all = pd_month(ds["valid_time"].values)

    # Day-of-season x positions for the daily figures. A wrapping window
    # needs two reference years so the axis runs forward across New Year.
    pos = []
    for m, d in slots:
        yr = (REF_YEAR_WRAP[1] if (wraps and (m, d) < args.season_start)
              else (REF_YEAR_WRAP[0] if wraps else REF_YEAR_START))
        pos.append(np.datetime64(f"{yr}-{m:02d}-{d:02d}"))
    pos = np.array(pos).astype("datetime64[D]").astype("O")

    n_seasons = len(seasons)
    return Analysis(
        args=args, ds=ds, idx=idx, slot=slot, syear=syear, slots=slots,
        seasons=seasons, n_seasons=n_seasons, wraps=wraps,
        cloudy=cloudy, liquid=liquid, liquid_any=liquid_any, valid=valid,
        w_yx=w_yx, n_cells=n_cells, daily=daily, monthly=monthly, med=med,
        months=months, dim=dim, pooled=pooled, month_of_all=month_of_all,
        pos=pos, x=np.arange(len(months)),
        names=[calendar.month_abbr[m] for m in months],
        years_label=_season_span(seasons),
        thr_label=(f"liquid = LWP > {args.lwp_threshold:g} g m$^{{-2}}$;  "
                   + _cloud_criterion_label(args.min_cloud_fraction)),
        band_note=("shading = 25th\u201375th percentile across seasons"
                   if n_seasons > 1 else None),
    )


def print_report(A):
    """The numeric summary that the command-line run prints."""
    args = A.args
    med = A.med
    daily = A.daily
    months = A.months
    slots = A.slots
    idx = A.idx
    cloudy = A.cloudy
    liquid = A.liquid
    liquid_any = A.liquid_any
    valid = A.valid
    w_yx = A.w_yx
    ds = A.ds
    pooled = A.pooled
    n_cells = A.n_cells
    # ---- (1) overall -------------------------------------------------------
    w = np.broadcast_to(w_yx, cloudy[idx].shape)
    c_w = float((cloudy[idx] * w).sum())
    l_w = float((liquid[idx] * w).sum())
    la_w = float((liquid_any[idx] * w).sum())
    v_w = float((valid[idx] * w).sum())
    n_cloudy, n_liquid, n_valid = (int(cloudy[idx].sum()), int(liquid[idx].sum()),
                                   int(valid[idx].sum()))
    print("\n" + "-" * 72)
    print("  (1) OVERALL, whole record, accepted seasons only")
    print("-" * 72)
    print(f"    cell-hours considered           : {n_valid * HOURS_PER_STEP:>14,.0f}")
    print(f"    cell-hours cloudy               : {n_cloudy * HOURS_PER_STEP:>14,.0f}"
          f"   ({100 * c_w / v_w:.2f}% by area)")
    print(f"    cell-hours cloudy + liquid      : {n_liquid * HOURS_PER_STEP:>14,.0f}"
          f"   ({100 * l_w / v_w:.2f}% by area)")
    print(f"    cell-hours liquid, ANY cloud    : {int(liquid_any[idx].sum()):>14,}"
          f"   ({100 * la_w / v_w:.2f}% by area)")
    print(f"    fraction of cloudy scenes with liquid : {100 * l_w / c_w:6.2f}%")
    print(f"    mean hours per cell, cloudy     : {n_cloudy / n_cells:>8,.0f} h")
    print(f"    mean hours per cell, liquid     : {n_liquid / n_cells:>8,.0f} h")

    print("\n    Sensitivity to --lwp-threshold (fraction of cloudy scenes):")
    print(f"      {'g m-2':>9}{'kg m-2':>10}{'fraction':>11}")
    tclw_a_g = ds["tclw"].values * 1000.0
    tciw_a_g = ds["tciw"].values * 1000.0
    tcc_a = ds["tcc"].values
    for thr in SENSITIVITY_THRESHOLDS:
        cl = valid & (tcc_a >= args.min_cloud_fraction) & \
             ((tclw_a_g > thr) | (tciw_a_g > args.iwp_threshold))
        li = cl & (tclw_a_g > thr)
        cw = float((cl[idx] * w).sum()); lw = float((li[idx] * w).sum())
        f = lw / cw if cw > 0 else np.nan
        mark = "  <- current" if abs(thr - args.lwp_threshold) < 1e-12 else ""
        print(f"      {thr:>9g}{thr/1000:>10g}{100*f:>10.2f}%{mark}")

    # ---- (2) monthly -------------------------------------------------------
    print("\n" + "-" * 72)
    print("  (2) MONTHLY, median across seasons")
    print("-" * 72)
    hdr = (f"    {'month':<7}{'cloudy h':>10}{'liquid h':>10}{'frac':>8}"
           f"{'LWP':>8}{'IWP':>8}{'snow':>8}{'rain':>8}{'total':>8}"
           f"{'TCWV':>8}{'t2m C':>8}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))

    def _col(key, j, scale=1.0, default=float("nan")):
        return scale * med[key][j] if key in med else default

    for j, m in enumerate(months):
        lwp, iwp = _col("lwp_cloudy", j), _col("iwp_cloudy", j)
        snow, rain = _col("snow_cloudy", j), _col("rain_cloudy", j)
        total = np.nansum([lwp, iwp, snow, rain])
        print(f"    {calendar.month_abbr[m]:<7}{med['cloudy_h'][j]:>10,.1f}"
              f"{med['liquid_h'][j]:>10,.1f}{100*med['frac'][j]:>7.2f}%"
              f"{lwp:>8.1f}{iwp:>8.1f}{snow:>8.1f}{rain:>8.1f}{total:>8.1f}"
              f"{_col('tcwv_cloudy', j):>8.2f}"
              f"{_col('t2m_cloudy', j) - 273.15:>8.1f}")
    print("    paths in g m-2, TCWV in kg m-2, all conditioned on cloudy scenes.")
    print("    'snow' and 'rain' are the PRECIPITATING condensate (tcsw, tcrw);")
    print("    ERA5's IWP (tciw) is suspended cloud ice only and excludes them.")

    # ---- (3) daily ---------------------------------------------------------
    print("\n" + "-" * 72)
    print("  (3) DAILY, median across seasons")
    print("-" * 72)
    d_med = {k: med_iqr(v)[0] for k, v in daily.items()}
    print(f"    days in season            : {len(slots)}")
    print(f"    mean cloudy hours per day : {np.nanmean(d_med['cloudy_h']):5.2f} of 24")
    print(f"    mean liquid hours per day : {np.nanmean(d_med['liquid_h']):5.2f} of 24")
    print(f"    liquid ANY cloud, per day : {np.nanmean(d_med['liquid_any_h']):5.2f} of 24")

    # ---- (4) per cell-day --------------------------------------------------
    print("\n" + "-" * 72)
    print("  (4) PER CELL-DAY, quantiles over all cells x all seasons")
    print("-" * 72)
    print("    hours per day in a typical cell-day, median (25th-75th):")
    for name, lab in (("cloudy", "overcast          "),
                      ("cloud_liquid", "overcast + liquid "),
                      ("liquid_any", "any liquid aloft  ")):
        q = pooled[name]
        print(f"    {lab}: {np.nanmean(q[:, 1]):5.2f} "
              f"({np.nanmean(q[:, 0]):5.2f}-{np.nanmean(q[:, 2]):5.2f}) of 24")



def condensate_series(A):
    """Which condensate paths to draw, and in what colour.

    Shared by the two condensate figures so they cannot disagree about whether
    the precipitating species are included.
    """
    series = [("lwp_cloudy", LIQUID_COLOR, "LWP (liquid)"),
              ("iwp_cloudy", ICE_COLOR, "IWP (cloud ice)")]
    if A.args.include_precip:
        series += [(k, c, l) for k, c, l in
                   (("snow_cloudy", SNOW_COLOR, "snow water"),
                    ("rain_cloudy", RAIN_COLOR, "rain water")) if k in A.med]
    return series


def fig_monthly_liquid(A, out_dir=None, dpi=None):
    """fig 1: monthly hours + fraction.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    med = A.med
    years_label = A.years_label
    thr_label = A.thr_label
    x = A.x
    names = A.names
    # --- fig 1: monthly hours + fraction ------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    a1.bar(x - 0.2, med["cloudy_h"], width=0.4, color=CLOUD_COLOR, label="cloudy")
    a1.bar(x + 0.2, med["liquid_h"], width=0.4, color=LIQUID_COLOR,
           label="cloudy + liquid")
    a1.set_xticks(x); a1.set_xticklabels(names)
    a1.legend(fontsize=9, framealpha=0.9)
    a1.set_title(f"Cloud liquid frequency \u2014 {args.region}\n"
                 f"{years_label}   |   {thr_label}", fontsize=12, pad=10)
    _finish(a1, "Hours per cell\nper month")
    a2.bar(x, 100 * med["frac"], width=0.55, color=LIQUID_COLOR)
    a2.set_xticks(x); a2.set_xticklabels(names)
    _finish(a2, "Cloudy scenes with\nliquid [%]", (0, 100))

    return _emit(fig, A, out_dir, "cloud_liquid_monthly", dpi)


def fig_daily_liquid(A, out_dir=None, dpi=None):
    """fig 2: daily hours + fraction, with IQR.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    daily = A.daily
    years_label = A.years_label
    thr_label = A.thr_label
    n_seasons = A.n_seasons
    band_note = A.band_note
    pos = A.pos
    # --- fig 2: daily hours + fraction, with IQR ----------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                 constrained_layout=True)
    for key, color, lab in (("cloudy_h", CLOUD_COLOR, "cloudy"),
                            ("liquid_h", LIQUID_COLOR, "cloudy + liquid")):
        m50, m25, m75 = med_iqr(daily[key])
        if n_seasons > 1:
            a1.fill_between(pos, m25, m75, color=color, alpha=0.20, linewidth=0)
        a1.plot(pos, m50, color=color, lw=1.5, label=f"{lab} (median)")
    a1.legend(fontsize=9, framealpha=0.9, loc="lower left", title=band_note,
              title_fontsize=8)
    a1.set_title(f"Cloud liquid frequency by day of season \u2014 {args.region}\n"
                 f"{years_label}   |   {thr_label}", fontsize=12, pad=10)
    _finish(a1, "Hours per cell\nper day (of 24)", (0, 24))
    f50, f25, f75 = med_iqr(daily["frac"])
    if n_seasons > 1:
        a2.fill_between(pos, 100*f25, 100*f75, color=LIQUID_COLOR, alpha=0.20,
                        linewidth=0)
    a2.plot(pos, 100*f50, color=LIQUID_COLOR, lw=1.5)
    _finish(a2, "Cloudy scenes with\nliquid [%]", (0, 100))
    a2.set_xlabel("Day of season", fontsize=10); _daily_axis(a2)

    return _emit(fig, A, out_dir, "cloud_liquid_daily", dpi)


def fig_condensate_monthly(A, out_dir=None, dpi=None):
    """fig 3: monthly mean LWP and IWP.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    med = A.med
    years_label = A.years_label
    x = A.x
    names = A.names
    # --- fig 3: monthly mean LWP and IWP ------------------------------------
    # With --include-precip the falling species are shown too. Without them the
    # figure understates how much frozen water the winter column holds, because
    # the IFS carries prognostic snow: tciw is suspended cloud ice ONLY.
    series = condensate_series(A)
    fig, ax = plt.subplots(figsize=(11, 5.0), constrained_layout=True)
    n_b = len(series)
    width = 0.8 / n_b
    for i, (key, color, lab) in enumerate(series):
        off = (i - (n_b - 1) / 2) * width
        ax.bar(x + off, med[key], width=width * 0.92, color=color, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(f"Mean condensate path in cloudy scenes \u2014 {args.region}\n"
                 f"{years_label}   |   {_cloud_criterion_label(args.min_cloud_fraction)}",
                 fontsize=12, pad=10)
    _finish(ax, "Mean path [g m$^{-2}$]")

    return _emit(fig, A, out_dir, "cloud_condensate_monthly", dpi)


def fig_condensate_daily(A, out_dir=None, dpi=None):
    """fig 4: daily mean LWP and IWP when cloudy.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    daily = A.daily
    years_label = A.years_label
    n_seasons = A.n_seasons
    band_note = A.band_note
    pos = A.pos
    # --- fig 4: daily mean LWP and IWP when cloudy --------------------------
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    for key, color, lab in condensate_series(A):
        m50, m25, m75 = med_iqr(daily[key])
        if n_seasons > 1:
            ax.fill_between(pos, m25, m75, color=color, alpha=0.20,
                            linewidth=0)
        ax.plot(pos, m50, color=color, lw=1.5, label=f"{lab} (median)")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right", title=band_note,
              title_fontsize=8)
    ax.set_title(f"Daily mean condensate path in cloudy scenes \u2014 {args.region}\n"
                 f"{years_label}   |   {_cloud_criterion_label(args.min_cloud_fraction)}",
                 fontsize=12, pad=10)
    _finish(ax, "Mean path [g m$^{-2}$]")
    ax.set_xlabel("Day of season", fontsize=10); _daily_axis(ax)

    return _emit(fig, A, out_dir, "cloud_condensate_daily", dpi)


def fig_liquid_hours(A, out_dir=None, dpi=None):
    """fig 5: hours per cell-day with a liquid-bearing cloud.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    pooled = A.pooled
    years_label = A.years_label
    thr_label = A.thr_label
    pos = A.pos
    # --- fig 5: hours per cell-day with a liquid-bearing cloud --------------
    # The sample unit here is a cell-day, not a season-day: for "Aug 01" the
    # quantiles run over every grid cell on every 1 August in the record. That
    # is the question "in a typical cell, for how many of the 24 hours was a
    # liquid-bearing cloud overhead?", which an area mean cannot answer -- an
    # area mean of 12 h is produced equally by every cell being cloudy half the
    # day and by half the strip being cloudy all day.
    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    q = pooled["cloud_liquid"]
    ax.fill_between(pos, q[:, 0], q[:, 2], color=LIQUID_COLOR, alpha=0.20,
                    linewidth=0)
    ax.plot(pos, q[:, 1], color=LIQUID_COLOR, lw=1.6,
            label="overcast + liquid (median)")
    ax.plot(pos, pooled["liquid_any"][:, 1], color=LIQUID_COLOR, lw=1.1,
            ls="--", alpha=0.85, label="any liquid aloft, no cloud test (median)")
    ax.axhline(24, color="0.6", lw=0.8, ls=":")
    ax.legend(fontsize=9, framealpha=0.9, loc="lower left",
              title="shading = 25th\u201375th percentile over cells \u00d7 seasons",
              title_fontsize=8)
    ax.set_title(f"Hours per day with a liquid-bearing cloud \u2014 {args.region}\n"
                 f"{years_label}   |   {thr_label}   |   per cell-day, from 24 "
                 f"instantaneous hourly samples", fontsize=12, pad=10)
    _finish(ax, "Hourly samples with\nliquid-bearing cloud (of 24)", (0, 24.8))
    ax.set_xlabel("Day of season", fontsize=10); _daily_axis(ax)

    return _emit(fig, A, out_dir, "cloud_liquid_hours", dpi)


def fig_lwp_histogram(A, out_dir=None, dpi=None):
    """fig 6: per-month LWP histogram.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    slots = A.slots
    seasons = A.seasons
    idx = A.idx
    slot = A.slot
    syear = A.syear
    cloudy = A.cloudy
    valid = A.valid
    w_yx = A.w_yx
    ds = A.ds
    dim = A.dim
    month_of_all = A.month_of_all
    years_label = A.years_label
    # --- fig 6: per-month LWP histogram -------------------------------------
    edges = lwp_bin_edges(args.lwp_bin_min, args.lwp_bin_max, args.lwp_bins)
    h_months, H = monthly_lwp_histogram(ds, idx, slot, syear, seasons, slots,
                                        cloudy, valid, w_yx, edges, dim,
                                        args.lwp_threshold)
    # Mean across seasons, not median: bar heights then add up to the month's
    # mean cloudy hours, which a per-bin median would not (the median of a sum
    # is not the sum of medians).
    hist = nanmean_quiet(H, axis=0)                        # [month, bin]

    n_m = len(h_months)
    if args.hist_layout:
        n_r, n_c = args.hist_layout
        if n_r * n_c < n_m:
            print(f"  Note: --hist-layout {n_r}x{n_c} holds {n_r*n_c} panels but "
                  f"the season spans {n_m} months; widening to fit.",
                  file=sys.stderr)
            n_c = -(-n_m // n_r)
    else:
        n_r = 2
        n_c = -(-n_m // n_r)          # ceil, so Aug-Mar gives 2 x 4 not 2 x 3
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.4 * n_c, 3.6 * n_r),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.09)
    axes = np.atleast_1d(axes).ravel()
    n_bar = len(edges) + 2
    centres = np.arange(n_bar)
    # Bars are categorical, so the decade labels have to be placed by bin index
    # rather than by value. The top edge gets no tick of its own: the overflow
    # bar sits beside it and its ">" label already names that edge.
    ticks = [0, 1]
    labels = ["0" if args.lwp_threshold <= 0 else f"$\\leq${args.lwp_threshold:g}",
              f"{edges[0]:g}"]
    for i, e in enumerate(edges[1:-1], start=1):
        if abs(np.log10(e) - round(np.log10(e))) < 1e-9:
            # Bar i+1 holds (edges[i-1], edges[i]], so edge i sits at its left face.
            ticks.append(i + 0.5); labels.append(f"{e:g}")
    ticks.append(n_bar - 1); labels.append(f">{edges[-1]:g}")
    colors = [ICE_COLOR] + [LIQUID_COLOR] * (n_bar - 1)

    med_lwp = monthly_median_lwp(ds, idx, month_of_all, cloudy, w_yx, h_months,
                                 args.lwp_threshold)
    print("\n    Median LWP over cloudy scenes with liquid present (LWP > threshold):")
    for m in h_months:
        print(f"      {calendar.month_abbr[m]:<5}{med_lwp[m]:>9.2f} g m-2")

    for k, m in enumerate(h_months):
        ax = axes[k]
        ax.bar(centres, hist[k], width=0.9, color=colors)
        ax.set_title(f"{calendar.month_name[m]}   "
                     f"({np.nansum(hist[k]):,.0f} h cloudy)", fontsize=10, pad=6)
        xm = lwp_to_bar_x(med_lwp[m], edges)
        if xm is not None:
            ax.axvline(xm, color="#B2182B", lw=1.6, ls=":",
                       label=f"Median LWP, liquid only = {med_lwp[m]:,.3g} g m$^{{-2}}$")
            ax.legend(fontsize=8.5, framealpha=0.85, loc="best")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=11, rotation=45, ha="right")
        # sharex hides the top row's labels, which leaves the bottom row's
        # titles floating under the top row's bars and reading as its x labels.
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelsize=11)
    for ax in axes[n_m:]:
        ax.set_visible(False)
    for k in range(0, len(axes), n_c):
        if axes[k].get_visible():
            axes[k].set_ylabel("Hours per cell\nper month", fontsize=13)
    for k in range(len(axes) - n_c, len(axes)):
        if axes[k].get_visible():
            axes[k].set_xlabel("LWP [g m$^{-2}$]", fontsize=13)
    fig.suptitle(f"Cloudy-scene hours by liquid water path — {args.region}\n"
                 f"{years_label}, mean across seasons   |   "
                 f"{_cloud_criterion_label(args.min_cloud_fraction)}   |   "
                 f"log-spaced bins; leftmost bar = overcast with "
                 f"LWP $\\leq$ {args.lwp_threshold:g} g m$^{{-2}}$",
                 fontsize=12)

    return _emit(fig, A, out_dir, "cloud_lwp_histogram", dpi)


def fig_episode_duration(A, out_dir=None, dpi=None):
    """fig 7: episode durations vs the hourly sampling interval.

    Draws from a prepared :class:`Analysis`, so it can be re-run on its
    own without reloading the archive. Saves into ``out_dir`` when one is
    given, and always returns the figure.
    """
    import matplotlib.pyplot as plt
    args = A.args
    med = A.med
    months = A.months
    idx = A.idx
    cloudy = A.cloudy
    liquid = A.liquid
    valid = A.valid
    w_yx = A.w_yx
    ds = A.ds
    month_of_all = A.month_of_all
    years_label = A.years_label
    # --- fig 7: episode durations vs the hourly sampling interval -----------
    # The point of this figure is a methodological one: tcc is an INSTANTANEOUS
    # field, so counting hourly samples measures duration only if episodes last
    # many hours. That is testable, and this is the test.
    epi = episode_statistics(ds["valid_time"].values, idx,
                             {"cloudy": cloudy, "liquid": liquid},
                             w_yx, month_of_all)

    print("\n" + "-" * 72)
    print("  (5) EPISODE DURATION vs the 1-hour sampling interval")
    print("-" * 72)
    for name, lab in (("cloudy", "overcast"), ("liquid", "overcast + liquid")):
        rec = epi[name]
        tot = rec["hist"].sum(axis=0)
        mean, med, one_h = run_summary(tot)
        cens = 100 * rec["censored"] / max(rec["total"], 1)
        hours = tot * np.arange(tot.size)
        share_1 = 100 * hours[1] / hours.sum()
        share_le2 = 100 * hours[1:3].sum() / hours.sum()
        print(f"    {lab:<18} mean {mean:5.2f} h   median {med:4.0f} h"
              f"   episodes lasting exactly 1 h: {100*one_h:5.1f}%")
        print(f"    {'':<18} share of total {lab} TIME in episodes of"
              f" 1 h: {share_1:4.1f}%,  <=2 h: {share_le2:4.1f}%")
        print(f"    {'':<18} episodes discarded as censored at a record gap:"
              f" {cens:.1f}%")

    if args.episode_scan:
        # How much of the 1-hour spike is physical, and how much is cells
        # wobbling across an exact tcc threshold? Relaxing the threshold
        # separates the two. The memoryless column is the share of 1-hour
        # episodes a geometric process with the SAME measured persistence would
        # produce; the excess over it is real short-timescale structure.
        tcc_a = ds["tcc"].values
        tclw_a_g = ds["tclw"].values * 1000.0
        tciw_a_g = ds["tciw"].values * 1000.0
        print("\n    Sensitivity of episode duration to --min-cloud-fraction:")
        print(f"      {'tcc >=':>7}{'base rate':>11}{'mean h':>9}{'median h':>10}"
              f"{'1-h eps':>9}{'memoryless':>12}{'time in 1-h':>13}")
        for thr in EPISODE_SCAN_THRESHOLDS:
            m = valid & (tcc_a >= thr) & ((tclw_a_g > args.lwp_threshold) |
                                          (tciw_a_g > args.iwp_threshold))
            rec = episode_statistics(ds["valid_time"].values, idx,
                                     {"m": m}, w_yx, month_of_all)["m"]
            tot = rec["hist"].sum(axis=0)
            mean, med, one = run_summary(tot)
            hours = tot * np.arange(tot.size)
            pair = rec["pair_on"].sum()
            p_stay = rec["pair_on_on"].sum() / pair if pair > 0 else np.nan
            mark = "  <- current" if abs(thr - args.min_cloud_fraction) < 1e-12 else ""
            print(f"      {thr:>7.2f}{m[idx].mean():>11.3f}{mean:>9.2f}{med:>10.0f}"
                  f"{100*one:>8.1f}%{100*(1-p_stay):>11.1f}%"
                  f"{100*hours[1]/hours.sum():>12.1f}%{mark}")

    n_top = 24
    fig, ((b1, b2), (b3, b4)) = plt.subplots(2, 2, figsize=(13.5, 8.0),
                                             constrained_layout=True)

    # (a) how long episodes last
    xs = np.arange(1, n_top + 2)
    for i, (name, color, lab) in enumerate(
            (("cloudy", CLOUD_COLOR, "overcast"),
             ("liquid", LIQUID_COLOR, "overcast + liquid"))):
        tot = epi[name]["hist"].sum(axis=0)
        frac = np.empty(n_top + 1)
        frac[:n_top] = tot[1:n_top + 1]
        frac[n_top] = tot[n_top + 1:].sum()
        frac = 100 * frac / tot[1:].sum()
        b1.bar(xs + (i - 0.5) * 0.42, frac, width=0.42, color=color, label=lab)
    # Memoryless benchmark. If the overcast state had no memory beyond one hour
    # -- if the chance of surviving the next hour were always p, no matter how
    # long the deck had already been there -- episode lengths would be geometric:
    # P(length = k) = p**(k-1) * (1 - p). Feeding in the SAME persistence p that
    # was measured from the data makes this a like-for-like reference, so any
    # departure is memory in the process, not a different overall cloudiness.
    rec = epi["cloudy"]
    p_stay = rec["pair_on_on"].sum() / rec["pair_on"].sum()
    g = p_stay ** (xs[:n_top] - 1) * (1 - p_stay)
    g = np.append(g, p_stay ** n_top)          # the >n_top tail
    b1.plot(xs, 100 * g, color="#B2182B", lw=1.5, ls="--", marker="o", ms=3,
            label=f"memoryless, same persistence (p = {p_stay:.3f})")
    b1.set_xlabel("Episode duration [h]", fontsize=9)
    b1.set_ylabel("Share of episodes [%]", fontsize=9)
    b1.set_xticks([1, 4, 8, 12, 16, 20, 24, 25])
    b1.set_xticklabels(["1", "4", "8", "12", "16", "20", "24", f">{n_top}"])
    b1.legend(fontsize=8.5, framealpha=0.9)
    b1.set_title("(a) Episode duration distribution", fontsize=10)

    # (b) where the TIME actually sits. Short episodes are numerous but carry
    # little of the record, which is the quantity that matters for aliasing.
    for name, color, lab in (("cloudy", CLOUD_COLOR, "overcast"),
                             ("liquid", LIQUID_COLOR, "overcast + liquid")):
        tot = epi[name]["hist"].sum(axis=0)
        hours = tot * np.arange(tot.size)
        cdf = 100 * np.cumsum(hours) / hours.sum()
        b2.step(np.arange(tot.size), cdf, where="post", color=color, lw=1.6,
                label=lab)
        b2.plot([1], [cdf[1]], "o", color=color, ms=5)
        b2.annotate(f"{lab}: {cdf[1]:.1f}% of time is in 1-h episodes",
                    (1, cdf[1]), textcoords="offset points",
                    xytext=(12, 2 if name == "cloudy" else 16),
                    fontsize=8, color=color)
    b2.axvline(1.0, color="#B2182B", lw=1.2, ls="--")
    b2.set_xlim(0, 48); b2.set_ylim(0, 100)
    b2.set_xlabel("Episode duration [h]", fontsize=9)
    b2.set_ylabel("Share of total time in episodes\nno longer than this [%]",
                  fontsize=9)
    b2.legend(fontsize=9, framealpha=0.9, loc="lower right")
    b2.set_title("(b) Cumulative share of overcast time", fontsize=10)

    # (c) duration by month
    order = [m for m in months]
    xm = np.arange(len(order))
    for i, (name, color, lab) in enumerate(
            (("cloudy", CLOUD_COLOR, "overcast"),
             ("liquid", LIQUID_COLOR, "overcast + liquid"))):
        means = [run_summary(epi[name]["hist"][m])[0] for m in order]
        b3.bar(xm + (i - 0.5) * 0.42, means, width=0.42, color=color, label=lab)
    b3.axhline(1.0, color="#B2182B", lw=1.2, ls="--")
    b3.text(len(order) - 0.4, 1.6, "1 h", color="#B2182B", fontsize=8, ha="right")
    b3.set_xticks(xm); b3.set_xticklabels([calendar.month_abbr[m] for m in order])
    b3.set_ylabel("Mean episode duration [h]", fontsize=9)
    b3.legend(fontsize=9, framealpha=0.9)
    b3.set_title("(c) Mean duration by month", fontsize=10)

    # (d) hour-to-hour persistence
    for i, (name, color, lab) in enumerate(
            (("cloudy", CLOUD_COLOR, "overcast"),
             ("liquid", LIQUID_COLOR, "overcast + liquid"))):
        rec = epi[name]
        pers = [rec["pair_on_on"][m] / rec["pair_on"][m]
                if rec["pair_on"][m] > 0 else np.nan for m in order]
        b4.bar(xm + (i - 0.5) * 0.42, pers, width=0.42, color=color, label=lab)
    b4.set_xticks(xm); b4.set_xticklabels([calendar.month_abbr[m] for m in order])
    b4.set_ylim(0, 1.0)
    b4.set_ylabel("P(state holds one hour later)", fontsize=9)
    b4.legend(fontsize=9, framealpha=0.9, loc="lower right")
    b4.set_title("(d) Hour-to-hour persistence", fontsize=10)

    for ax in (b1, b2, b3, b4):
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Cloud episode duration vs the hourly sampling interval "
                 f"\u2014 {args.region}\n{years_label}   |   "
                 f"{_cloud_criterion_label(args.min_cloud_fraction)}   |   "
                 f"episodes spanning a record gap are excluded", fontsize=12)

    return _emit(fig, A, out_dir, "cloud_episode_duration", dpi)


ALL_FIGURES = (
    fig_monthly_liquid,
    fig_daily_liquid,
    fig_condensate_monthly,
    fig_condensate_daily,
    fig_liquid_hours,
    fig_lwp_histogram,
    fig_episode_duration,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    import matplotlib.pyplot as plt
    print()
    for fn in ALL_FIGURES:
        fn(A, out_dir=out_dir)
    if args.show:
        plt.show()
    print("=" * 72)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_data_source_args(parser)
    parser.add_argument("--season-start", type=parse_month_day, default=(8, 1),
                        metavar="MM-DD")
    parser.add_argument("--season-end", type=parse_month_day, default=(3, 31),
                        metavar="MM-DD", help="Inclusive; may wrap the new year.")
    parser.add_argument("--years", type=parse_years, default=None, metavar="SPEC")
    parser.add_argument("--lwp-threshold", type=float, default=0.0, metavar="G",
                        help="g m-2. NOTE: 0 yields ~100%% because ERA5 carries "
                             "trace liquid nearly everywhere; try 10-25 (roughly "
                             "a microwave radiometer's detection floor).")
    parser.add_argument("--iwp-threshold", type=float, default=0.0, metavar="G",
                        help="g m-2, used in the cloudy test.")
    parser.add_argument("--min-cloud-fraction", type=float, default=1.0,
                        metavar="F")
    parser.add_argument("--min-season-coverage", type=float, default=0.5,
                        metavar="F",
                        help="Exclude seasons covering less than this fraction of "
                             "the window (default 0.5). Guards against a season "
                             "with a handful of days distorting the means.")
    parser.add_argument("--include-precip", action="store_true",
                        help="Add the precipitating condensate (tcsw snow water, "
                             "tcrw rain water) to the condensate figures. ERA5's "
                             "IWP is suspended cloud ice only, so without this "
                             "the frozen column water is badly understated in "
                             "winter, when most of it is falling snow.")
    parser.add_argument("--lwp-bin-min", type=float, default=0.1, metavar="G",
                        help="First LWP histogram edge, g m-2 (default 0.1). "
                             "Everything at or below it, including exactly zero, "
                             "goes in the first bar.")
    parser.add_argument("--lwp-bin-max", type=float, default=1000.0, metavar="G",
                        help="Last LWP histogram edge, g m-2 (default 1000).")
    parser.add_argument("--lwp-bins", type=int, default=12, metavar="N",
                        help="Log-spaced bins between the two edges (default 12, "
                             "i.e. 3 per decade over 0.1-1000).")
    parser.add_argument("--hist-layout", type=parse_layout, default=None,
                        metavar="RxC",
                        help="Panel grid for the LWP histogram, e.g. 2x4. "
                             "Default: 2 rows, enough columns for the months in "
                             "the season. Widened automatically if too small.")
    parser.add_argument("--episode-scan", action="store_true",
                        help="Also report how episode duration depends on the\n"
                             "cloud-cover threshold. Slow: it redoes the run\n"
                             "extraction once per threshold.")
    parser.add_argument("--ocean-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=350)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())

