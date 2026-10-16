#!/usr/bin/env python
"""Compare two ARM liquid-water-path retrievals at the Scripps Pier, season by season: MWRRET v2 vs MWRRET v1.

Why: the GOES-like LWP in the seasonal-averages table (check_seasonal_averages.ipynb, Section 3c) comes from
MWRRET v2, and its winter overcast mean is higher than summer's. MWRRET v1 uses a different radiometer and
a different retrieval, so it checks whether v2 behaves differently in winter. This script compares the two
directly, sample by sample, under low cloud.

The two retrievals (EPCAPE products in config.yaml; both files cover 15 Feb 2023 - 14 Feb 2024):
  MWRRET v1  lwp_mwrret1_M1 = epcmwrret1liljclouM1.c2 (Turner et al. 2007, IEEE TGRS 45, 3680-3690,
             doi:10.1109/TGRS.2007.903703). 2-channel MWR (23.8 and 31.4 GHz), one sample every ~20 s.
               phys_lwp              physical-iterative retrieval: optimal estimation with the monoRTM forward
                                     model and interpolated radiosonde profiles (file attribute
                                     physical_retrieval_comment).
               phys_lwp_uncertainty  1-sigma uncertainty of phys_lwp from the optimal estimation, assuming a
                                     0.25 K uncertainty in each brightness temperature (file attributes).
               be_lwp                "best estimate": equal to phys_lwp, except where phys_lwp has bad quality,
                                     where it is stat2_lwp. stat2 is a statistical retrieval whose coefficients
                                     are predicted from surface meteorology (file attributes). be_lwp is missing
                                     if both are bad, or if the value it would copy has Indeterminate quality.
               be_retrieval_status   which retrieval be_lwp holds: 0 none, 1 physical, 2 stat2.
  MWRRET v2  lwp_mwrret2_M1 = epcmwrret2turnM1.c1. 3-channel MWR (mwr3c), one sample every second.
             The 1-s stream is not continuous: the radiometer pauses for 14-15 s about every 75-90 s
             (its own duty cycle), and the record has a 22-day outage, 14 Oct - 6 Nov 2023, which falls
             in the Fall season. Figure 4 shows what this does to the pairing. (The average spacing over
             the year is 1.4 s, but that number is just the gaps averaged in; the instrument samples at 1 s.)
               phys_lwp              physical retrieval. The file has no uncertainty for it.

Figures (2 x 2, one panel per season; Spring = Apr-Jun 2023, Summer = Jul-Sep 2023, Fall = Oct-Dec 2023,
Winter = 15 Feb-31 Mar 2023 + 1 Jan-14 Feb 2024, i.e. the sheet's seasons, seasonal.SEASONS, UTC):
  1. y = v2 phys_lwp, x = v1 phys_lwp, with x error bars = v1 phys_lwp_uncertainty (1 sigma).
  2. the same, but x = v1 be_lwp. Points where be_lwp holds the stat2 retrieval are violet triangles. The
     file gives no uncertainty for stat2, so those points have no error bar. Physical points look as in
     figure 1.
  3. the pairs figures 1 and 2 are NOT allowed to use: the windows where some v2 samples passed QC and some
     did not (see "Filters" below). y is the mean of the samples that did pass, and each point is coloured
     by the share of its window that passed, so a point built from 2 of 21 samples is visibly paler than one
     built from 19 of 21. The axes are much wider than in figures 1 and 2 because these are nearly all
     high-LWP cases. Each panel also carries its figure-1 fit as a dotted reference line: that is the
     comparison to make, since the question is whether these rejected windows behave like the kept ones.
     Figure 3 exists because the rejected pairs are few (about 1,400 of 545,000) but are concentrated in
     exactly the high-LWP winter cases the seasonal-averages question is about, so they would be invisible
     if they were simply mixed into figure 1.
  4. how many v2 samples each window actually holds, season by season (a bar chart, not a scatter plot):
     the plotted pairs against every candidate window. This says how much averaging stands behind each v2
     value and how many windows hold nothing usable at all.
Panels 1-3 show the 1:1 line (thin black), an ordinary least-squares fit y = a x + b (dashed), and the
squared Pearson correlation r^2. The statistics use every paired sample, including any beyond the axes.

How the samples are paired (one point per v1 sample):
  * v2 value = the mean of the v2 samples within ±MATCH_HALF_WINDOW_S (10 s) of the v1 time stamp, i.e. a
    20-s window that matches v1's ~20-s sampling. The window is centred because the time-stamp convention is
    not documented in the files. Checked on 16 Oct 2026: moving the window by -20 to +20 s changes the
    campaign r^2 by at most 0.02 (0.746-0.767), so the alignment does not matter here.
  * How many v2 samples that is varies: about 21 when the window falls between two of the radiometer's
    pauses, about 7-8 when it contains one, and 0 in an outage. The v2 value is a 21-sample mean in some
    places and a 7-sample mean in others, which is worth remembering when reading the scatter. Figure 4
    shows the distribution, and v2_n in the paired table records it for every pair.
  * EVERY v2 sample in the window must pass QC (below) for the pair to be used. One bad sample out of 21
    rejects the whole window, so that no plotted v2 value mixes good and bad retrievals. This is strict on
    purpose, and it is not free: over the campaign it rejects about 1,400 pairs that still had good samples
    to average, and those are mostly high-LWP winter cases (median v1 LWP about 540 g m-2, against 73 g m-2
    for the pairs that are kept; 1,022 of them in winter). Figure 3 is there to show them rather than lose
    them. Far more pairs - about 57,000 - are lost because NO sample in the window passed, and nothing can
    be done about those: there is no v2 value to plot.
Filters (all must hold):
  * QC, "recommended for use" (the user's requirement):
      v2: qc_phys_lwp = 0 and phys_qc_flag = 0 (0 = retrievals_good). In this record the two flag exactly the
          same 3.3 % of samples (see quantities.lwp_qc_rain_free_gm2).
      v1, figure 1: qc_phys_lwp = 0. MWRRET v1 has NO phys_qc_flag; qc_phys_lwp is its only QC field for
          phys_lwp. Bits: 1 missing value (Bad); 2 cloud_base_height missing (Indeterminate; set for 34 % of
          samples, mostly clear sky); 3 brightness temperature failed max, water on window, warm-up/overheat
          or spike (Indeterminate); 4 brightness temperature missing, spike, invalid optical depth, min or
          > 200 K (Bad). Requiring 0 rejects all four, Indeterminate included (as everywhere in this project).
      v1, figure 2: qc_be_lwp = 0 and be_retrieval_status 1 or 2. phys_lwp's QC is NOT required here, because
          the point of be_lwp is to fall back to stat2 where phys_lwp is bad.
  * Low cloud overhead: every Pier ceilometer sample (epcceilM1.b1, 16 s) within ±MATCH_HALF_WINDOW_S of the
    v1 time stamp (at least one) has a detected cloud base with qc_first_cbh = 0 and first_cbh <=
    LOW_CLOUD_MAX_M (3500 m; detection_status 1-3), as in the GOES-like table
    (quantities.ceilometer_low_cloud_state). Fog without a detectable base (status 4) counts as unknown and
    is dropped.
  * Rain (optional; OFF by default): with --rain-exclude-min N, pairs within
    ±N minutes of rain at the Pier Parsivel (precip_rate > 0, QC = 0) are dropped (quantities.near_rain).
    A wet radome biases both radiometers high.
No LWP range limit is applied: negative cloudy-sky retrievals and very large (often rainy) values stay in.

Statistics per panel (also written to a CSV): n pairs; OLS slope and intercept; r^2; mean bias (v2 - v1);
RMSD; median v1 uncertainty; the number of points beyond the axes; and, in figure 2, the number and share of
stat2 points. How often be_lwp uses stat2 is also counted over ALL v1 samples with a valid be_lwp (any sky).
Two caveats on the fit:
  * v1 has measurement error, so the OLS slope is biased toward 0 by the ratio
    var(true x) / (var(true x) + sigma^2) (regression dilution; Frost & Thompson 2000, J. R. Stat. Soc. A
    163, 173-189). With sigma ~ 10 g m-2 against a spread of ~100 g m-2 this is ~1 %, far too small to
    explain slopes of 0.4-0.9. r^2 is symmetric in x and y and does not depend on the fit method.
  * The pairs are not independent of one another. They are 20 s apart in a cloud field that takes hours to
    change, so a panel of 100,000 pairs does not hold 100,000 independent pieces of information - closer to
    one per cloud event. This does not bias the slope, r^2, bias or RMSD, and nothing here quotes an
    uncertainty, so nothing is wrong as the script stands. It matters only if someone later puts an error
    bar or a p-value on one of these numbers: doing that needs an effective sample size (how many
    independent events there are), not n.

Every figure states in its title whether the rain screen was applied, and so do all file names.
Outputs (in mwr_lwp_comparison/, next to this script; --out chooses another folder). <tag> is _no_rain_screen
or _rain_screen_<N>min, plus _cbh<H>m for a low-cloud limit other than 3500 m, so no run overwrites another:
  MWRRET_v2_vs_v1_phys_lwp_by_season<tag>.png / .pdf            figure 1
  MWRRET_v2_vs_v1_be_lwp_by_season<tag>.png / .pdf              figure 2
  MWRRET_v2_vs_v1_partial_qc_windows_by_season<tag>.png / .pdf  figure 3
  MWRRET_v2_window_coverage_by_season<tag>.png / .pdf           figure 4
  MWRRET_v2_vs_v1_fit_statistics<tag>.csv              the panel statistics of figures 1, 2 and 3
  MWRRET_v1_be_lwp_stat2_usage<tag>.csv                how often be_lwp uses stat2, by season
  MWRRET_v2_window_coverage<tag>.csv                   the numbers behind figure 4, by season

Examples (from the EPCAPE folder)
  python comparisons/seasonal_averages/compare_MWRRET_v1_v2_LWP.py                        # no rain screen
  python comparisons/seasonal_averages/compare_MWRRET_v1_v2_LWP.py --rain-exclude-min 30  # ±30 min rain screen
  python comparisons/seasonal_averages/compare_MWRRET_v1_v2_LWP.py --axis-max-gm2 500
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from EPCAPE.analysis_tools import units  # noqa: E402
from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402
from EPCAPE.plotting import COLORS, GRID, INK, INK_2, INK_3, use_style  # noqa: E402

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
SEASONS = ("Spring", "Summer", "Fall", "Winter")  # panel order: (a) (b) on the top row, (c) (d) below
SEASON_LABELS = {"Spring": "Spring (Apr-Jun 2023)", "Summer": "Summer (Jul-Sep 2023)",
                 "Fall": "Fall (Oct-Dec 2023)", "Winter": "Winter (15 Feb-Mar 2023 + Jan-14 Feb 2024)"}
MATCH_HALF_WINDOW_S = 10.0  # v2 and ceilometer samples within ± this of a v1 time stamp are paired with it
LOW_CLOUD_MAX_M = 3500.0  # low cloud: ceilometer base at or below this height (--cbh-max-m overrides)
DEFAULT_LOW_CLOUD_MAX_M = LOW_CLOUD_MAX_M  # file names name the limit only when it differs from this
RAIN_EXCLUDE_MIN = 30.0  # 0 = no rain screen (default); N > 0 drops pairs within ±N min of Pier Parsivel rain
AXIS_MIN_GM2 = -50.0  # lower axis limit. Retrievals under cloud can come out slightly negative (noise around a
#                      small true LWP); v1 reaches -89 g m-2, so a few hundred points sit below this limit and
#                      are counted in the "beyond the axes" line rather than being shown. v2 stops at -32.
AXIS_MAX_GM2 = 900.0  # upper axis limit, fixed so that runs with and without the rain screen share axes (900 =
#                      the automatic limit of the unscreened record); 0 (--axis-max-gm2 0) = automatic: the
#                      99.5th percentile of all plotted values, rounded up to 100 (500 with the ±30-min screen)
PARTIAL_AXIS_MAX_GM2 = 2200.0  # figure 3 has its own, much larger limit: the windows it shows are the ones the
#                      QC rule rejects, and those are nearly all high-LWP cases that run far off the 900 axis.
#                      Fixed for the same reason as AXIS_MAX_GM2; --axis-max-gm2 0 makes this one automatic too.
AUTO_AXIS_PERCENTILE = 99.5
PHYS_COLOR = COLORS["MWR"]  # aqua, the project's microwave-radiometer hue: physical-retrieval points
STAT2_COLOR = "#4a3aa7"  # violet (categorical slot 7 of the reference palette): stat2 points. OKLab
#                          lightness 43 vs 67 for the aqua, so the two stay distinct under colour-vision
#                          deficiency; the triangle marker is a second cue.
FIT_COLOR = INK_2  # dashed OLS fit line, neutral ink
POINT_SIZE = 4.0  # marker area (pt^2) of the ~10^5 points per panel
POINT_ALPHA = 0.25
ERRORBAR_COLOR = "#0e6b4a"  # a darker shade of the aqua, so the short bars show through the translucent markers
ERRORBAR_ALPHA = 0.2  # ~10^5 overlapping bars: translucent, drawn behind the markers. sigma(v1) is ~9-13 g m-2
#                       at every LWP (checked 16 Oct 2026), ~1 % of the axis, so the bars are short by nature.
AXIS_LABEL_FONTSIZE = 13  # the style's 10 pt + 3 pt, as in investigating_high_wintertime_LWP_during_EPCAPE.py
# Figure 3 colours each point by how much of its window passed QC. That is a magnitude, not an identity, so
# it gets one hue running light to dark (never a rainbow: a rainbow has no order a reader can rank). The hue
# is the project's microwave aqua, so the figure still reads as an MWR figure.
COVERAGE_CMAP = LinearSegmentedColormap.from_list("mwr_coverage", ["#d8f3e8", PHYS_COLOR, "#0a4b35"])
CANDIDATE_COLOR = INK_3  # figure 4: the outlined bars for every candidate window (the paired ones are aqua)


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
def _ns(times) -> np.ndarray:
    """Time stamps as int64 nanoseconds (for fast searchsorted window look-ups)."""
    return np.asarray(times, dtype="datetime64[ns]").astype(np.int64)


def check_strictly_increasing(t_ns: np.ndarray, name: str) -> None:
    """Stop if a source time axis is not strictly increasing.

    ``window_bounds`` relies on ``np.searchsorted``, which assumes a sorted source: an unsorted or
    duplicated time axis silently returns the wrong window instead of raising, so every source the
    windows are taken from is checked here. (The rain screen needs no check: ``Q.near_rain`` sorts the
    rain time stamps itself.)"""
    if t_ns.size and np.any(np.diff(t_ns) <= 0):
        raise ValueError(f"{name} time stamps are not strictly increasing")


def window_bounds(t_ref_ns: np.ndarray, t_src_ns: np.ndarray, half_ns: int):
    """For each reference time, the [lo, hi) index range of the (sorted) source samples within ±half_ns."""
    lo = np.searchsorted(t_src_ns, t_ref_ns - half_ns, side="left")
    hi = np.searchsorted(t_src_ns, t_ref_ns + half_ns, side="right")
    return lo, hi


def paired_samples(cbh_max_m: float, rain_exclude_min: float) -> pd.DataFrame:
    """One row per MWRRET v1 sample, with the matched v2 mean, the cloud screen and every QC field.

    Columns (LWP in g m-2):
      v1_phys_lwp, v1_phys_sigma   v1 phys_lwp and its 1-sigma uncertainty, NaN where qc_ != 0
      v1_be_lwp, v1_be_status      v1 be_lwp (NaN where qc_be_lwp != 0) and be_retrieval_status (0/1/2)
      v2_lwp                       mean of the QC'd v2 samples in the window (NaN if none passes)
      v2_n, v2_n_ok                v2 samples in the window, and how many of them pass QC
      v2_all_ok                    True when the window has >= 1 v2 sample and all of them pass QC
      low_cloud                    True when the window has >= 1 ceilometer sample and all show low cloud
      rain_free                    True unless rain lies within ±rain_exclude_min (always True when 0)
      season                       one of SEASONS, or "" outside them
    """
    half_ns = int(MATCH_HALF_WINDOW_S * 1e9)

    # --- MWRRET v1: QC = 0 on each variable (Q.qc0_values sets every qc_<var> != 0 sample to NaN)
    v1 = load_product("lwp_mwrret1_M1")
    t1 = _ns(v1["time"].values)
    v1_phys = units.water_path_to_gm2(Q.qc0_values(v1, "phys_lwp")[0]).values
    v1_sigma = units.water_path_to_gm2(Q.qc0_values(v1, "phys_lwp_uncertainty")[0]).values
    v1_be = units.water_path_to_gm2(Q.qc0_values(v1, "be_lwp")[0]).values
    v1_status = v1["be_retrieval_status"].values

    # --- MWRRET v2: a sample passes when qc_phys_lwp = 0 AND phys_qc_flag = 0. Window sums come from
    # cumulative sums, so each v1 time costs two searchsorted look-ups (22 million v2 samples).
    v2 = load_product("lwp_mwrret2_M1")
    t2 = _ns(v2["time"].values)
    check_strictly_increasing(t2, "lwp_mwrret2_M1")
    v2_lwp = units.water_path_to_gm2(Q.qc0_values(v2, "phys_lwp")[0]).values.astype(float)
    flag = v2["phys_qc_flag"].values
    v2_ok = np.isfinite(v2_lwp) & np.isfinite(flag) & (flag == 0)
    cum_ok = np.concatenate([[0], np.cumsum(v2_ok)])  # running count of passing samples
    cum_lwp = np.concatenate([[0.0], np.cumsum(np.where(v2_ok, v2_lwp, 0.0))])  # running sum of their LWP
    lo, hi = window_bounds(t1, t2, half_ns)
    n_v2 = hi - lo  # all v2 samples in the window
    n_v2_ok = cum_ok[hi] - cum_ok[lo]  # the passing ones
    with np.errstate(invalid="ignore", divide="ignore"):
        v2_mean = (cum_lwp[hi] - cum_lwp[lo]) / n_v2_ok  # NaN when none passes

    # --- low cloud overhead from the Pier ceilometer: 1 low cloud, 0 none, NaN unknown (fog, failed QC)
    state = Q.ceilometer_low_cloud_state(load_product("cbh_ceil_M1"), max_cbh_m=cbh_max_m)
    tc = _ns(state.index.values)
    check_strictly_increasing(tc, "cbh_ceil_M1")
    cum_low = np.concatenate([[0], np.cumsum(state.to_numpy() == Q.LOW_CLOUD)])
    lo_c, hi_c = window_bounds(t1, tc, half_ns)
    n_ceil = hi_c - lo_c
    low_cloud = (n_ceil >= 1) & ((cum_low[hi_c] - cum_low[lo_c]) == n_ceil)  # every sample low cloud

    # --- optional rain screen at the Pier Parsivel (QC = 0), as in the GOES-like table
    times = pd.DatetimeIndex(v1["time"].values)
    if rain_exclude_min > 0:
        rain = Q.to_series(Q.qc0_values(load_product("rain_ld_M1"), "precip_rate")[0])
        rain_free = ~Q.near_rain(rain, times, window_min=rain_exclude_min)
    else:
        rain_free = np.ones(times.size, dtype=bool)

    season = np.full(times.size, "", dtype=object)
    for name in SEASONS:
        season[S.season_mask(times, name)] = name

    return pd.DataFrame(
        {"v1_phys_lwp": v1_phys, "v1_phys_sigma": v1_sigma, "v1_be_lwp": v1_be, "v1_be_status": v1_status,
         "v2_lwp": v2_mean, "v2_n": n_v2, "v2_n_ok": n_v2_ok,
         "v2_all_ok": (n_v2 >= 1) & (n_v2_ok == n_v2),
         "low_cloud": low_cloud, "rain_free": rain_free, "season": season},
        index=times,
    )


def selection(pairs: pd.DataFrame, x_col: str, *, v2_qc: str = "all") -> pd.DataFrame:
    """The pairs a figure plots: both retrievals usable, low cloud overhead, (rain-free), in a season.

    x_col "v1_phys_lwp": figures 1 and 3 (v1 qc_phys_lwp = 0, and its uncertainty QC = 0 too).
    x_col "v1_be_lwp":   figure 2 (v1 qc_be_lwp = 0 and be_retrieval_status 1 or 2).

    v2_qc says which windows to take, where a "window" is the ±MATCH_HALF_WINDOW_S of v2 samples around
    one v1 time stamp (about 21 samples, v2 running at 1 s):
      "all"      every v2 sample in the window passes QC. Figures 1 and 2; the strict rule.
      "partial"  some pass and some do not. These are exactly the windows "all" throws away while a
                 usable v2 mean could still be formed from the samples that passed, so figure 3 can
                 show what is being lost. A window where NO sample passes has no v2 value at all and
                 is in neither group.
    """
    if v2_qc == "all":
        keep = pairs["v2_all_ok"].copy()
    elif v2_qc == "partial":
        keep = ~pairs["v2_all_ok"]  # the v2_lwp test below then requires >= 1 sample to have passed
    else:
        raise ValueError(f"v2_qc must be 'all' or 'partial', not {v2_qc!r}")
    keep &= pairs["low_cloud"] & pairs["rain_free"] & (pairs["season"] != "")
    keep &= np.isfinite(pairs[x_col]) & np.isfinite(pairs["v2_lwp"])
    if x_col == "v1_phys_lwp":
        keep &= np.isfinite(pairs["v1_phys_sigma"])
    elif x_col == "v1_be_lwp":
        keep &= pairs["v1_be_status"].isin([1, 2])
    else:
        raise ValueError(f"x_col must be 'v1_phys_lwp' or 'v1_be_lwp', not {x_col!r}")
    return pairs[keep]


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def fit_statistics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """OLS fit y = slope x + intercept, r^2 (squared Pearson r), mean bias (y - x) and RMSD, in g m-2.

    A panel can end up with fewer than two pairs (a narrow --cbh-max-m, a long --rain-exclude-min, a
    short record): a straight line through fewer than two points is undefined, and numpy does not
    return NaN for it but raises (np.polyfit: TypeError for no point, LinAlgError for one). Those
    panels therefore get NaN for everything the fit defines, and the per-sample statistics only where
    they still mean something (one pair still has a bias and a median; none has neither)."""
    n = int(x.size)
    if n < 2:
        diff = y - x
        return {"n": n, "slope": np.nan, "intercept_gm2": np.nan, "r2": np.nan,
                "bias_v2_minus_v1_gm2": float(diff.mean()) if n else np.nan,
                "rmsd_gm2": float(np.sqrt(np.mean(diff ** 2))) if n else np.nan,
                "median_v1_gm2": float(np.median(x)) if n else np.nan,
                "median_v2_gm2": float(np.median(y)) if n else np.nan}
    slope, intercept = np.polyfit(x, y, 1)  # least squares, y on x
    r = np.corrcoef(x, y)[0, 1]
    diff = y - x
    return {"n": n, "slope": slope, "intercept_gm2": intercept, "r2": r ** 2,
            "bias_v2_minus_v1_gm2": diff.mean(), "rmsd_gm2": np.sqrt(np.mean(diff ** 2)),
            "median_v1_gm2": np.median(x), "median_v2_gm2": np.median(y)}


def stat2_usage(pairs: pd.DataFrame, plotted: pd.DataFrame) -> pd.DataFrame:
    """How often be_lwp holds the stat2 retrieval, by season (EPCAPE = all four seasons together).

    all_valid_*: every v1 sample with a usable be_lwp - qc_be_lwp = 0 and be_retrieval_status 1 or 2 - at
                 any sky, with no cloud, v2 or rain filter. (The status test costs nothing: in this record
                 status 0, "no retrieval", never comes with qc_be_lwp = 0. It is kept so the column means
                 "a real retrieval is in there" whatever a future file does.)
    plotted_*:   the pairs of figure 2 (low cloud, v2 QC'd, ...).
    """
    rows = []
    for name in ("EPCAPE",) + SEASONS:
        in_s = (pairs["season"] != "") if name == "EPCAPE" else (pairs["season"] == name)
        valid = in_s & np.isfinite(pairs["v1_be_lwp"]) & pairs["v1_be_status"].isin([1, 2])
        n_stat2 = int((valid & (pairs["v1_be_status"] == 2)).sum())
        p_in = plotted if name == "EPCAPE" else plotted[plotted["season"] == name]
        p_stat2 = int((p_in["v1_be_status"] == 2).sum())
        rows.append({"season": name,
                     "all_samples": int(in_s.sum()),
                     "status_0_no_retrieval": int((in_s & (pairs["v1_be_status"] == 0)).sum()),
                     "all_valid_be_lwp": int(valid.sum()), "all_valid_stat2": n_stat2,
                     "all_valid_stat2_pct": 100.0 * n_stat2 / max(int(valid.sum()), 1),
                     "plotted_pairs": int(len(p_in)), "plotted_stat2": p_stat2,
                     "plotted_stat2_pct": 100.0 * p_stat2 / max(len(p_in), 1)})
    return pd.DataFrame(rows).set_index("season")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def axis_max_gm2(selections) -> float:
    """Common upper axis limit: AUTO_AXIS_PERCENTILE of max(x, y) over every plotted pair, rounded up to 100.

    With no pair at all there is no percentile to take (np.percentile raises on an empty array), so the
    empty figures fall back to the fixed default rather than stopping the run."""
    both = np.concatenate([np.maximum(sel["x"].to_numpy(), sel["y"].to_numpy()) for sel in selections]
                          + [np.empty(0)])
    if both.size == 0:
        print(f"  no pairs to scale the axes with: keeping the default upper limit ({AXIS_MAX_GM2:g} g m-2)")
        return float(AXIS_MAX_GM2)
    return float(np.ceil(np.percentile(both, AUTO_AXIS_PERCENTILE) / 100.0) * 100.0)


def plot_by_season(sel: pd.DataFrame, *, x_label: str, title: str, subtitle: str, footer: str, ax_max: float,
                   show_stat2: bool, color_by_coverage: bool = False, reference_fits=None,
                   extra_lines=None, bottom: float = 0.18) -> tuple:
    """2 x 2 scatter plots of v2 (y) vs v1 (x), one panel per season. Returns (figure, statistics table).

    sel columns: x, y, sigma (x error, NaN = no bar), status (be_retrieval_status; used if show_stat2), season,
    and, for figure 3 only, coverage (the share of the window that passed QC, 0-1).
    subtitle: the line under the title (the rain-screen statement, rain_screen_title).

    The last four arguments are figure 3's: colour the points by coverage and add a colour bar; draw each
    panel's figure-1 fit as a faint reference line, so it is clear whether these windows follow the same
    relation as the ones the strict rule keeps; add per-season lines to the statistics box; and leave more
    room at the bottom for the colour bar.
    """
    use_style()
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 12.4), sharex=True, sharey=True)
    lims = (AXIS_MIN_GM2, ax_max)
    stats = []
    for ax, name, letter in zip(axes.flat, SEASONS, "abcd"):
        d = sel[sel["season"] == name]
        x, y, sig = d["x"].to_numpy(), d["y"].to_numpy(), d["sigma"].to_numpy()
        is_stat2 = (d["status"] == 2).to_numpy() if show_stat2 else np.zeros(x.size, dtype=bool)
        phys = ~is_stat2

        # error bars first (faint, behind), then the physical-retrieval points, then stat2 on top
        has_bar = phys & np.isfinite(sig)
        ax.errorbar(x[has_bar], y[has_bar], xerr=sig[has_bar], fmt="none", ecolor=ERRORBAR_COLOR,
                    elinewidth=0.6, alpha=ERRORBAR_ALPHA, capsize=0, zorder=1, rasterized=True)
        if color_by_coverage:
            # few points (these are the rejected windows), so they are drawn large, opaque and with a thin
            # white ring that keeps overlapping markers apart
            ax.scatter(x, y, c=d["coverage"].to_numpy(), cmap=COVERAGE_CMAP, vmin=0.0, vmax=1.0, s=22,
                       alpha=0.9, linewidths=0.3, edgecolors="white", zorder=3, rasterized=True)
        else:
            ax.scatter(x[phys], y[phys], s=POINT_SIZE, color=PHYS_COLOR, alpha=POINT_ALPHA, linewidths=0,
                       zorder=2, rasterized=True)
        if is_stat2.any():
            ax.scatter(x[is_stat2], y[is_stat2], s=16, marker="^", color=STAT2_COLOR, alpha=0.9,
                       linewidths=0, zorder=4, rasterized=True)

        # 1:1 line (thin black) and the OLS fit over every pair in the panel (including any beyond the axes)
        st = fit_statistics(x, y)
        ax.plot(lims, lims, color=INK, lw=0.8, zorder=5)
        xf = np.array(lims)
        if reference_fits is not None and name in reference_fits:
            r_slope, r_intercept = reference_fits[name]
            if np.isfinite(r_slope):
                ax.plot(xf, r_slope * xf + r_intercept, color=INK_3, lw=1.1, ls=(0, (1, 2)), zorder=5)
        if np.isfinite(st["slope"]):  # fewer than two pairs: no fit to draw (see fit_statistics)
            ax.plot(xf, st["slope"] * xf + st["intercept_gm2"], color=FIT_COLOR, lw=1.3, ls="--", zorder=5)

        # statistics box in the upper left, where v2 > v1 by a lot (nearly empty)
        n_off = int(((x > ax_max) | (y > ax_max) | (x < lims[0]) | (y < lims[0])).sum())
        lines = [f"n = {st['n']:,} pairs"]
        if np.isfinite(st["slope"]):
            lines += [f"fit: y = {st['slope']:.2f} x {'+' if st['intercept_gm2'] >= 0 else '−'} "
                      f"{abs(st['intercept_gm2']):.1f}",
                      f"r² = {st['r2']:.3f}"]
        else:
            lines.append("fit: too few pairs")
        if st["n"]:
            lines += [f"mean(v2 − v1) = {st['bias_v2_minus_v1_gm2']:+.1f} g m⁻²",
                      f"RMSD = {st['rmsd_gm2']:.1f} g m⁻²",
                      f"median v1 / v2 = {st['median_v1_gm2']:.0f} / {st['median_v2_gm2']:.0f} g m⁻²"]
        if np.isfinite(sig).any():
            lines.append(f"median σ(v1) = {np.nanmedian(sig):.1f} g m⁻²")
        if show_stat2:
            stat2_pct = 100.0 * is_stat2.mean() if x.size else 0.0
            lines.append(f"stat2 points: {int(is_stat2.sum()):,} ({stat2_pct:.2f} %)")
        if color_by_coverage and x.size:
            lines.append(f"median window passing QC = {100.0 * d['coverage'].median():.0f} %")
        lines.append(f"beyond the axes: {n_off:,}")
        if extra_lines is not None:
            lines += list(extra_lines.get(name, []))
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                color=INK, zorder=6, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.6))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"({letter}) {SEASON_LABELS[name]}", loc="left")
        st.update({"season": name, "median_v1_sigma_gm2": float(np.nanmedian(sig)) if np.isfinite(sig).any()
                   else np.nan, "n_stat2": int(is_stat2.sum()), "n_beyond_axes": n_off,
                   "median_window_qc_pass_frac": float(d["coverage"].median()) if color_by_coverage
                   and x.size else np.nan})
        stats.append(st)

    for ax in axes[1, :]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel("MWRRET v2 phys_lwp (3-channel MWR) [g m⁻²]")

    # one legend for the whole figure, in a row above the panels, so it never covers data
    handles = []
    if not color_by_coverage:  # in figure 3 the colour bar is the legend for the markers
        handles.append(Line2D([], [], ls="none", marker="o", ms=5, color=PHYS_COLOR,
                              label="v1 physical retrieval (bars: ±1σ phys_lwp_uncertainty)"))
    if show_stat2:
        handles.append(Line2D([], [], ls="none", marker="^", ms=6, color=STAT2_COLOR,
                              label="v1 stat2 retrieval (no uncertainty in the file)"))
    handles += [Line2D([], [], color=INK, lw=0.8, label="1:1"),
                Line2D([], [], color=FIT_COLOR, lw=1.3, ls="--", label="least-squares fit")]
    if reference_fits is not None:
        handles.append(Line2D([], [], color=INK_3, lw=1.1, ls=(0, (1, 2)),
                              label="fit to the strictly QC'd pairs (figure 1)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=3 if len(handles) == 3 else 2, fontsize=9.5, frameon=False)
    fig.suptitle(title, y=0.988, fontsize=12.5, fontweight="semibold")
    fig.text(0.5, 0.958, subtitle, ha="center", va="center", fontsize=11, color=INK)
    fig.text(0.06, 0.012, footer, ha="left", va="bottom", fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.875, bottom=bottom, wspace=0.08, hspace=0.12)
    if color_by_coverage:
        # a horizontal colour bar in the bottom margin, under the x label and above the footer. `bottom`
        # has to leave room for all three: panels end at `bottom`, the x label sits just under them, the
        # bar and its label take the next ~0.05, and the footer grows upward from 0.012.
        cax = fig.add_axes((0.32, bottom - 0.075, 0.36, 0.011))
        bar = fig.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0), cmap=COVERAGE_CMAP), cax=cax,
                           orientation="horizontal")
        bar.set_label("share of the v2 window that passed QC", fontsize=9.5, color=INK)
        bar.ax.tick_params(labelsize=8.5, colors=INK_2, length=2)
        bar.outline.set_edgecolor(GRID)
    stats_df = pd.DataFrame(stats).set_index("season")
    return fig, stats_df


def plot_window_coverage(pairs: pd.DataFrame, plotted: pd.DataFrame, *, title: str, subtitle: str,
                         footer: str) -> tuple:
    """2 x 2 bar charts of how many v2 samples each v1 window holds, one panel per season.

    Two distributions per panel, because they answer two different questions:
      bars (aqua)   the pairs figure 1 plots. Every sample in those windows passed QC, so the count is
                    also the number of good v2 retrievals that went into the plotted v2 value.
      line (grey)   every candidate window (v1 QC = 0, low cloud, rain screen if on, in a season),
                    counting only the samples that passed QC. This one includes 0, i.e. the windows
                    with nothing usable in them, which is the single largest loss of pairs.
    Both are drawn as a percentage of their own group, so seasons of different length compare directly.

    Returns (figure, table). `pairs` is the full frame from ``paired_samples``; `plotted` is the
    figure-1 selection.
    """
    use_style()
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    candidate = (np.isfinite(pairs["v1_phys_lwp"]) & pairs["low_cloud"] & pairs["rain_free"]
                 & (pairs["season"] != ""))
    cand = pairs[candidate]
    n_max = int(max(cand["v2_n_ok"].max(), plotted["v2_n"].max(), 1))
    bins = np.arange(n_max + 1)  # 0 ... n_max samples in a window

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), sharex=True, sharey=True)
    rows = []
    for ax, name, letter in zip(axes.flat, SEASONS, "abcd"):
        p_n = plotted.loc[plotted["season"] == name, "v2_n"].to_numpy()
        c_n = cand.loc[cand["season"] == name, "v2_n_ok"].to_numpy()
        p_pct = 100.0 * np.bincount(p_n, minlength=n_max + 1)[: n_max + 1] / max(p_n.size, 1)
        c_pct = 100.0 * np.bincount(c_n, minlength=n_max + 1)[: n_max + 1] / max(c_n.size, 1)
        ax.bar(bins, p_pct, width=0.8, color=PHYS_COLOR, alpha=0.85, linewidth=0, zorder=3)
        ax.step(bins, c_pct, where="mid", color=CANDIDATE_COLOR, lw=1.2, zorder=4)

        med = float(np.median(p_n)) if p_n.size else np.nan
        pct_le10 = 100.0 * float((p_n <= 10).mean()) if p_n.size else np.nan
        pct_none = 100.0 * float((c_n == 0).mean()) if c_n.size else np.nan
        ax.text(0.03, 0.97, "\n".join([f"pairs plotted: {p_n.size:,}",
                                       f"median samples per window: {med:.0f}",
                                       f"windows with ≤ 10 samples: {pct_le10:.0f} %",
                                       f"candidate windows: {c_n.size:,}",
                                       f"... with no QC'd v2 sample: {pct_none:.0f} %"]),
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.6))
        ax.set_title(f"({letter}) {SEASON_LABELS[name]}", loc="left")
        ax.set_xlim(-0.8, n_max + 0.8)
        rows.append({"season": name, "pairs_plotted": int(p_n.size), "median_samples_per_window": med,
                     "pct_pairs_le_10_samples": pct_le10,
                     "pct_pairs_full_window": 100.0 * float((p_n >= n_max - 1).mean()) if p_n.size else np.nan,
                     "candidate_windows": int(c_n.size), "pct_candidates_no_qc_sample": pct_none})

    for ax in axes[1, :]:
        ax.set_xlabel(f"v2 samples in the ±{MATCH_HALF_WINDOW_S:g} s window")
    for ax in axes[:, 0]:
        # short, and smaller than the x label: these panels are not tall enough for a long rotated label
        ax.set_ylabel("share of the group [%]", fontsize=11)
    handles = [Line2D([], [], ls="none", marker="s", ms=7, color=PHYS_COLOR,
                      label="pairs plotted in figure 1 (all samples passed QC)"),
               Line2D([], [], color=CANDIDATE_COLOR, lw=1.2,
                      label="all candidate windows, QC'd samples only (0 included)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2, fontsize=9.5,
               frameon=False)
    fig.suptitle(title, y=0.983, fontsize=12.5, fontweight="semibold")
    fig.text(0.5, 0.942, subtitle, ha="center", va="center", fontsize=11, color=INK)
    fig.text(0.06, 0.012, footer, ha="left", va="bottom", fontsize=8.5, color=INK_2, linespacing=1.4)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.835, bottom=0.21, wspace=0.08, hspace=0.30)
    return fig, pd.DataFrame(rows).set_index("season")


def rain_screen_title(rain_exclude_min: float) -> str:
    """The figure subtitle that says whether the rain screen was applied."""
    if rain_exclude_min > 0:
        return f"Rain screen ON: pairs within ±{rain_exclude_min:g} min of Pier Parsivel rain removed"
    return "No rain screen: pairs near rain included (a wet radome biases both radiometers high)"


def rain_screen_tag(rain_exclude_min: float) -> str:
    """The file-name part that says whether the rain screen was applied."""
    return f"_rain_screen_{rain_exclude_min:g}min" if rain_exclude_min > 0 else "_no_rain_screen"


FOOTER_WIDTH = 135  # characters that fit across the 10.5-inch figure at 8.5 pt, with a margin


def wrap_footer(lines) -> str:
    """One footer string from a list of sentences, each wrapped so it cannot run off the page."""
    return "\n".join(textwrap.fill(line, width=FOOTER_WIDTH, subsequent_indent="    ") for line in lines)


def footer_text(cbh_max_m: float, rain_exclude_min: float, v1_rule: str, *, v2_rule: str = "") -> str:
    """The filters, in a few lines under the panels."""
    rain = (f"no Pier Parsivel rain within ±{rain_exclude_min:g} min." if rain_exclude_min > 0
            else "no rain screen (wet-radome samples included).")
    v2_rule = v2_rule or (
        "a pair is kept only if EVERY v2 sample in its window has qc_phys_lwp = 0 AND phys_qc_flag = 0. "
        "v2 runs at 1 s, so that is about 21 samples, and ONE bad sample rejects the whole window.")
    return wrap_footer([
        f"Pairs: one per MWRRET v1 sample (~20 s); v2 = mean of its 1-s samples within "
        f"±{MATCH_HALF_WINDOW_S:g} s of the v1 time stamp.",
        f"QC, v2: {v2_rule}",
        f"QC, v1: {v1_rule}",
        f"Low cloud: every Pier ceilometer sample within ±{MATCH_HALF_WINDOW_S:g} s has a base "
        f"(detection_status 1-3, qc_first_cbh = 0) ≤ {cbh_max_m:g} m.",
        f"Rain: {rain}",
        "Fit: ordinary least squares, y on x, over all pairs in the panel (also those beyond the axes); "
        "r² = squared Pearson correlation.",
        # "The pairs are 20 s apart in a cloud field that changes over hours, so they are not independent "
        # "samples. Quote the slope, r², bias and RMSD as they are, but do not put an error bar or a p-value "
        # "on them without first accounting for that.",
        "The pairs are 20 s apart in a cloud field that changes over hours, so not all samples are independent.",
    ])


def save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rain-exclude-min", type=float, default=RAIN_EXCLUDE_MIN,
                        help="drop pairs within ± this many minutes of Pier Parsivel rain (default 0 = off)")
    parser.add_argument("--cbh-max-m", type=float, default=LOW_CLOUD_MAX_M,
                        help="low cloud = ceilometer base at or below this height (default 3500 m)")
    parser.add_argument("--axis-max-gm2", type=float, default=AXIS_MAX_GM2,
                        help="upper axis limit in g m-2 for figures 1 and 2 (default 900; figure 3 keeps "
                             "its own, wider limit). 0 = scale both to the data: the 99.5th percentile of "
                             "the plotted values, rounded up to 100")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "mwr_lwp_comparison",
                        help="output folder (default: mwr_lwp_comparison/, next to this script)")
    args = parser.parse_args()
    # 0 means "scale the axes to the data"; a negative limit would silently do the same, with no sign
    # of it in the figures or the file names, so it is rejected rather than interpreted.
    if args.axis_max_gm2 < 0:
        parser.error("--axis-max-gm2 must be 0 (automatic) or a positive upper limit in g m-2")
    args.out.mkdir(parents=True, exist_ok=True)

    tag = rain_screen_tag(args.rain_exclude_min)  # always names the rain screen, on or off
    if args.cbh_max_m != DEFAULT_LOW_CLOUD_MAX_M:
        tag += f"_cbh{args.cbh_max_m:g}m"

    print("Pairing MWRRET v1 and v2 samples ...")
    pairs = paired_samples(args.cbh_max_m, args.rain_exclude_min)

    # --- the two selections, in a common layout: x, y, sigma, status, season
    sel_phys = selection(pairs, "v1_phys_lwp")
    sel_phys = pd.DataFrame({"x": sel_phys["v1_phys_lwp"], "y": sel_phys["v2_lwp"],
                             "sigma": sel_phys["v1_phys_sigma"], "status": sel_phys["v1_be_status"],
                             "season": sel_phys["season"]})
    sel_be_raw = selection(pairs, "v1_be_lwp")
    # error bars only where be_lwp IS the physical retrieval (status 1, then be_lwp == phys_lwp exactly);
    # the file has no uncertainty for stat2
    sigma_be = sel_be_raw["v1_phys_sigma"].where(sel_be_raw["v1_be_status"] == 1)
    sel_be = pd.DataFrame({"x": sel_be_raw["v1_be_lwp"], "y": sel_be_raw["v2_lwp"], "sigma": sigma_be,
                           "status": sel_be_raw["v1_be_status"], "season": sel_be_raw["season"]})

    # figure 3's pairs: the windows the strict "every sample must pass" rule throws away even though some
    # of their v2 samples did pass. v2_lwp is already the mean of just those samples.
    sel_part_raw = selection(pairs, "v1_phys_lwp", v2_qc="partial")
    sel_part = pd.DataFrame({"x": sel_part_raw["v1_phys_lwp"], "y": sel_part_raw["v2_lwp"],
                             "sigma": sel_part_raw["v1_phys_sigma"], "status": sel_part_raw["v1_be_status"],
                             "season": sel_part_raw["season"],
                             "coverage": sel_part_raw["v2_n_ok"] / sel_part_raw["v2_n"]})

    ax_max = args.axis_max_gm2 if args.axis_max_gm2 > 0 else axis_max_gm2([sel_phys, sel_be])
    part_ax_max = PARTIAL_AXIS_MAX_GM2 if args.axis_max_gm2 > 0 else axis_max_gm2([sel_part])
    print(f"  axis limits: {AXIS_MIN_GM2:g} to {ax_max:g} g m-2 "
          f"(figures 1 and 2), {AXIS_MIN_GM2:g} to {part_ax_max:g} g m-2 (figure 3)")

    # --- figure 1: phys_lwp vs phys_lwp
    fig, st_phys = plot_by_season(
        sel_phys, x_label="MWRRET v1 phys_lwp (2-channel MWR) [g m⁻²]",
        title="MWRRET v2 vs v1 physical-retrieval LWP under low cloud, by season",
        subtitle=rain_screen_title(args.rain_exclude_min),
        footer=footer_text(args.cbh_max_m, args.rain_exclude_min,
                           "qc_phys_lwp = 0 (v1 has no phys_qc_flag; this rejects Bad and Indeterminate bits)."),
        ax_max=ax_max, show_stat2=False)
    save(fig, args.out, f"MWRRET_v2_vs_v1_phys_lwp_by_season{tag}")

    # --- figure 2: phys_lwp vs be_lwp, stat2 points highlighted
    fig, st_be = plot_by_season(
        sel_be, x_label="MWRRET v1 be_lwp (2-channel MWR) [g m⁻²]",
        title="MWRRET v2 physical-retrieval LWP vs v1 best-estimate LWP under low cloud, by season",
        subtitle=rain_screen_title(args.rain_exclude_min),
        footer=footer_text(args.cbh_max_m, args.rain_exclude_min,
                           "qc_be_lwp = 0 and be_retrieval_status 1 (physical) or 2 (stat2)."),
        ax_max=ax_max, show_stat2=True)
    save(fig, args.out, f"MWRRET_v2_vs_v1_be_lwp_by_season{tag}")

    # --- figure 3: the windows the strict v2 QC rule rejects, which are mostly high-LWP winter cases.
    # Each panel also carries its figure-1 fit, so the two populations can be compared by eye, and a count
    # of the windows that had nothing usable in them at all (those cannot be plotted: they have no v2 value).
    candidate = (np.isfinite(pairs["v1_phys_lwp"]) & pairs["low_cloud"] & pairs["rain_free"]
                 & (pairs["season"] != ""))
    extra = {}
    for name in SEASONS:
        in_s = candidate & (pairs["season"] == name)
        extra[name] = [f"kept by the strict rule: {int((in_s & pairs['v2_all_ok']).sum()):,}",
                       f"no QC'd v2 sample at all: {int((in_s & (pairs['v2_n_ok'] == 0)).sum()):,}"]
    fig, st_part = plot_by_season(
        sel_part, x_label="MWRRET v1 phys_lwp (2-channel MWR) [g m⁻²]",
        title="The pairs the strict v2 QC rule rejects: windows where only some v2 samples passed",
        subtitle=rain_screen_title(args.rain_exclude_min),
        footer=footer_text(
            args.cbh_max_m, args.rain_exclude_min,
            "qc_phys_lwp = 0 (v1 has no phys_qc_flag; this rejects Bad and Indeterminate bits).",
            v2_rule="the OPPOSITE of figures 1 and 2. Shown here are the windows where at least one v2 "
                    "sample passed qc_phys_lwp = 0 and phys_qc_flag = 0 but at least one did not, so the "
                    "strict rule drops them. y is the mean of the samples that DID pass; the colour says "
                    "what share of the window that was. Note the wider axes: these are mostly high-LWP "
                    "cases."),
        ax_max=part_ax_max, show_stat2=False, color_by_coverage=True, extra_lines=extra, bottom=0.26,
        reference_fits={s: (r["slope"], r["intercept_gm2"]) for s, r in st_phys.iterrows()})
    save(fig, args.out, f"MWRRET_v2_vs_v1_partial_qc_windows_by_season{tag}")

    # --- figure 4: how many v2 samples a window actually holds, season by season
    fig, coverage = plot_window_coverage(
        pairs, selection(pairs, "v1_phys_lwp"),
        title="How many v2 samples fall in each 20-s window, by season",
        subtitle=rain_screen_title(args.rain_exclude_min),
        footer=wrap_footer([
            f"Each v1 sample is paired with the v2 samples within ±{MATCH_HALF_WINDOW_S:g} s of it. v2 "
            f"samples at 1 s, so a full window holds about 21.",
            "The two peaks are the radiometer's own duty cycle, not missing data: v2 samples every 1 s but "
            "pauses for 14-15 s every 75-90 s (checked 16 Oct 2026), so a window either falls between two "
            "pauses (~20-21 samples) or contains one (~7-8).",
            "This matters because a v2 value built from 7 samples is noisier than one built from 21, and "
            "the panels show how often that happens.",
            "The two distributions differ almost only at 0: the windows with nothing usable in them are "
            "where the pairs are actually lost, not the partly-QC'd ones (those are figure 3).",
        ]))
    save(fig, args.out, f"MWRRET_v2_window_coverage_by_season{tag}")

    # --- tables
    stats = pd.concat({"phys_lwp (fig 1)": st_phys, "be_lwp (fig 2)": st_be,
                       "phys_lwp, rejected windows (fig 3)": st_part}, names=["v1 variable", "season"])
    stats.to_csv(args.out / f"MWRRET_v2_vs_v1_fit_statistics{tag}.csv", float_format="%.4g")
    usage = stat2_usage(pairs, sel_be_raw)
    usage.to_csv(args.out / f"MWRRET_v1_be_lwp_stat2_usage{tag}.csv", float_format="%.4g")
    coverage.to_csv(args.out / f"MWRRET_v2_window_coverage{tag}.csv", float_format="%.4g")

    pd.set_option("display.width", 200)
    cols = ["n", "slope", "intercept_gm2", "r2", "bias_v2_minus_v1_gm2", "rmsd_gm2", "median_v1_gm2",
            "median_v2_gm2", "median_v1_sigma_gm2", "n_stat2", "median_window_qc_pass_frac",
            "n_beyond_axes"]
    print("\nFit statistics (g m-2):")
    print(stats[cols].round(3).to_string())
    print("\nHow often v1 be_lwp holds the stat2 retrieval:")
    print(usage.round(3).to_string())
    print("\nv2 samples per 20-s window:")
    print(coverage.round(2).to_string())


if __name__ == "__main__":
    main()
