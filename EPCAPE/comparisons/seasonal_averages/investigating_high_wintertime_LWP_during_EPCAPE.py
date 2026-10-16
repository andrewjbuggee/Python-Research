#!/usr/bin/env python
"""Why is the winter GOES-like MWR LWP so high for overcast scenes? Diagnostic plots, and summer for comparison.

The seasonal-averages check (check_seasonal_averages.ipynb, Section 3c) gives a higher overcast LWP in winter
(99 ± 55 g m-2 daytime) than in summer (67 ± 53). This script uses EXACTLY the filters and the hourly
downsampling of that table and shows how LWP, cloud boundaries and the nearness of rain change as the hourly
low cloud fraction (the "overcast" criterion) is raised from 0.01 to 1:
  * a winter figure (3 panels: LWP, cloud base and top, time to rain);
  * a summer-vs-winter figure: the same 3 panels for summer (left column) and winter (right column), with the
    same y axis in each row so the two seasons compare directly.
Each figure is made twice: daytime hours only, and all hours ("all time").

How the MWR LWP becomes one value per hour (identical to Section 3c, "hour" sampling; the helpers are the same
functions in quantities.py = Q):
  1. MWRRET v2 phys_lwp (epcmwrret2turnM1.c1, the 3-channel MWR mwr3c, ~1.3-s samples), qc_phys_lwp = 0 and
     phys_qc_flag = 0 (Q.lwp_qc_rain_free_gm2), i.e. only retrievals the VAP recommends for use: qc_phys_lwp
     bit 1 = "recommended it not be used", phys_qc_flag 1 = "retrievals_bad". In this record the two fields
     flag the same 730,969 of 22,047,667 samples (3.3 %), so either one alone would give the same result.
  2. Rain screen: samples within ±RAIN_EXCLUDE_MIN minutes of rain are dropped (a wet radome biases the MWR
     high). "Rain" = any 1-min sample with a rain rate > 0 (any accumulation; Q.near_rain), from the gauges
     chosen by RAIN_GAUGES.
  3. 10 <= LWP <= 250 g m-2 for each sample (Q.goes_like_samples, sampling "hour"; Seethala's GOES-18 limits).
  4. Hourly LWP = the MEAN of the hour's kept samples (60-min windows on the UTC hour), as in the table.
Which hours count:
  5. Low cloud fraction of the hour = the fraction of the Pier ceilometer's 16-s samples with a cloud base
     overhead at or below LOW_CLOUD_MAX_M (detection_status 1-3 and qc_first_cbh = 0; clear = status 0 or 5,
     or a QC'd base above the limit), needing data for >= CF_MIN_COVERAGE of the hour
     (Q.ceilometer_low_cloud_state, Q.low_cloud_fraction). An hour enters the statistics at threshold x when
     its fraction is >= x (x = 0.95 is the table's "overcast").
  6. Daytime = solar zenith angle at the middle of the hour <= SZA_MAX_DEG (SZA_SOURCE: NOAA equations or
     RADFLUX cosine_zenith); all time = any hour.
  7. Seasons = the sheet's seasons (seasonal.SEASONS, UTC): Summer = 1 Jul - 30 Sep 2023; Winter = 15 Feb -
     31 Mar 2023 and 1 Jan - 14 Feb 2024.
The statistics: each hour is one value (its mean); every curve is the MEDIAN and the 25-75 % range of those
hourly values over the hours with low cloud fraction >= x. The table reports the MEAN ± std of the same hourly
values (printed by this script at x = 0.95 as a check). n = number of hours.

Panels (letters follow the panels shown):
  LWP              median and 25-75 % of the hourly LWP vs x, for 30 thresholds x between 0.01 and 1;
  LWP box plots    (only with --box-panel / SHOW_BOX_PANEL) the hourly LWP for x = 0.1, 0.2, ..., 1.0
                   (box = 25-75 %, line = median, whiskers = 1.5 x the box height (Tukey), dots beyond);
  cloud boundaries median cloud base (Pier ceilometer: the hour's mean of its low-cloud bases, the samples
                   behind the cloud fraction) and median cloud top (KAZR-ARSCL: the hour's mean top of the
                   lowest hydrometeor layer whose base is <= LOW_CLOUD_MAX_M; no limit on the top) vs x;
  time to rain     median and 25-75 % of the hourly mean time from the kept MWR samples to the CLOSEST rain
                   (before or after) vs x; with --prev-next-rain / SHOW_PREV_NEXT_RAIN also thin lines for the
                   time since the previous and until the next rain. The rain screen sets the floor at
                   RAIN_EXCLUDE_MIN.

Outputs (winter_LWP_investigation/, next to this script; --out chooses another folder; <tag> = pier_parsivel_<N>min
         or any_gauge_<N>min, plus _cbh<H>m for a low-cloud limit other than 3500 m, so no choice overwrites another):
  winter_LWP_vs_low_cloud_fraction_<daytime|alltime>_<tag>.png / .pdf           the winter figure
  summer_vs_winter_LWP_vs_low_cloud_fraction_<daytime|alltime>_<tag>.png / .pdf the comparison figure
  <season>_LWP_hourly_<tag>.csv          one row per hour with LWP: every quantity plotted
  <season>_LWP_by_threshold_<daytime|alltime>_<tag>.csv   the curves, by threshold x

Examples (from the EPCAPE folder; the settings below are the defaults)
  python comparisons/seasonal_averages/investigating_high_wintertime_LWP_during_EPCAPE.py
  python comparisons/seasonal_averages/investigating_high_wintertime_LWP_during_EPCAPE.py --rain-gauges any
  python comparisons/seasonal_averages/investigating_high_wintertime_LWP_during_EPCAPE.py --rain-exclude-min 180
  python comparisons/seasonal_averages/investigating_high_wintertime_LWP_during_EPCAPE.py --box-panel
  python comparisons/seasonal_averages/investigating_high_wintertime_LWP_during_EPCAPE.py --prev-next-rain
  python comparisons/seasonal_averages/investigating_high_wintertime_LWP_during_EPCAPE.py --cbh-max-m 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# EPCAPE/ is itself the Python package: put its parent folder on sys.path
# (this file is EPCAPE/comparisons/seasonal_averages/<script>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt  # noqa: E402

from EPCAPE.analysis_tools.products import load_product  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import quantities as Q  # noqa: E402
from EPCAPE.comparisons.seasonal_averages import seasonal as S  # noqa: E402
from EPCAPE.plotting import COLORS, GRID, INK, INK_2, INK_3, plain_log_axis, use_style  # noqa: E402

# ---------------------------------------------------------------------------
# settings (the defaults match check_seasonal_averages.ipynb, Section 3c)
# ---------------------------------------------------------------------------
RAIN_GAUGES = "pier_parsivel"  # "pier_parsivel": Pier Parsivel only (as the table);
#                                "any": Pier Parsivel + Mt. Soledad Parsivel + Pier 2D video disdrometer
RAIN_EXCLUDE_MIN = 30.0  # drop MWR samples within ± this many minutes of rain (table: 30)
LWP_RANGE_GM2 = (10.0, 250.0)  # keep MWR samples in this range (Seethala's GOES-18 limits)
LOW_CLOUD_MAX_M = 3500.0  # "low cloud": ceilometer base (and ARSCL lowest-layer base) at or below this height;
#                           --cbh-max-m overrides it (the table uses 3500 m)
DEFAULT_LOW_CLOUD_MAX_M = LOW_CLOUD_MAX_M  # file names name the limit only when it differs from this
WINDOW_MIN = 60.0  # one value per hour
CF_MIN_COVERAGE = 0.5  # an hour's cloud fraction needs ceilometer data for >= 50 % of the hour
SZA_MAX_DEG = 80.0  # daytime: solar zenith angle at mid-hour <= 80 deg
SZA_SOURCE = "noaa"  # "noaa" (Q.solar_zenith_deg) or "radflux" (RADFLUX cosine_zenith), as in the notebook
OVERCAST_CF = 0.95  # the table's "overcast" threshold, marked on the plots
SEASON = "Winter"  # the single-season figure
COMPARE_SEASONS = ("Summer", "Winter")  # the comparison figure: left column, right column
SEASON_LABELS = {"Spring": "Spring (Apr-Jun 2023)", "Summer": "Summer (Jul-Sep 2023)",
                 "Fall": "Fall (Oct-Dec 2023)", "Winter": "Winter (15 Feb-Mar 2023 + Jan-14 Feb 2024)"}
CF_THRESHOLDS = np.linspace(0.01, 1.0, 30)  # hours with low cloud fraction >= each value
SHOW_BOX_PANEL = False  # True (or --box-panel): add the LWP box-plot panel to both figures
SHOW_PREV_NEXT_RAIN = False  # True (or --prev-next-rain): time-to-rain panels also show the medians of the time
#                              since the previous rain and until the next rain (default: closest rain only)
BOX_THRESHOLDS = np.round(np.arange(0.1, 1.0001, 0.1), 1)  # x values of the box plots
HOURLY_COLUMNS = ("lwp_gm2", "cbh_m", "cloud_top_m", "closest_h", "prev_h", "next_h")  # hourly means
LWP_COLOR = COLORS["MWR"]  # the microwave-radiometer hue of the project palette
AXIS_LABEL_FONTSIZE = 13  # x and y axis labels: the style's 10 pt + 3 pt


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def rain_series(gauges: str) -> pd.Series:
    """1-min rain rates (mm h-1) of the chosen gauges, stacked (only times with rate > 0 matter downstream).

    Pier Parsivel: precip_rate with qc_precip_rate = 0 (as the notebook's pier_rain). Mt. Soledad Parsivel:
    the same. VDIS: rain_rate (no qc_ field; NaN = no drops, which never counts as rain)."""
    parts = [Q.to_series(Q.qc0_values(load_product("rain_ld_M1"), "precip_rate")[0])]
    if gauges == "any":
        parts.append(Q.to_series(Q.qc0_values(load_product("rain_ld_S2"), "precip_rate")[0]))
        parts.append(load_product("rain_vdis_M1")["rain_rate"].to_series())
    elif gauges != "pier_parsivel":
        raise ValueError(f"rain gauges must be 'pier_parsivel' or 'any', not {gauges!r}")
    return pd.concat(parts).sort_index()


def hours_to_rain(times: pd.DatetimeIndex, rain_times: np.ndarray) -> pd.DataFrame:
    """Hours from each time to the previous rain, to the next rain, and to the closest of the two.

    rain_times: sorted datetime64[ns] of every 1-min sample with rain (rate > 0). No rain before (after) a
    time in the record gives NaN for 'previous' ('next'), so it drops out of the hourly means."""
    t = times.values.astype("datetime64[ns]")
    k = np.searchsorted(rain_times, t)  # first rain at or after t
    one_h = np.timedelta64(1, "h")
    prev_h = np.where(k > 0, (t - rain_times[np.maximum(k - 1, 0)]) / one_h, np.nan)
    next_h = np.where(k < rain_times.size, (rain_times[np.minimum(k, rain_times.size - 1)] - t) / one_h, np.nan)
    return pd.DataFrame({"prev_h": prev_h, "next_h": next_h, "closest_h": np.fmin(prev_h, next_h)}, index=times)


def season_slices(season: str):
    """[(first day, last day + 1)] of a sheet season, for slicing time series."""
    return [(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)) for a, b in S.SEASONS[season]]


def hourly_tables(seasons: Sequence[str], rain_gauges: str, rain_exclude_min: float,
                  sza_source: str) -> Dict[str, pd.DataFrame]:
    """For each season: one row per hour holding at least one kept MWR sample, with the hourly MEANS the
    panels need. The data are read once and shared by all seasons."""
    hour = pd.Timedelta(minutes=WINDOW_MIN)

    # --- MWR LWP samples: QC, rain screen, LWP range (the notebook's "hour" sampling) 
    # load LWP MWRRET v2 phys_lwp, QC = 0, rain-free, 10 <= LWP <= 250 g m-2 per sample, downsample to one value per hour
    # -------------------
    rain = rain_series(rain_gauges)
    ds = load_product("lwp_mwrret2_M1")
    lwp_all, _ = Q.lwp_qc_rain_free_gm2(ds, rain, rain_window_min=rain_exclude_min)
    del ds  # ~22 million samples
    lwp_all, _ = Q.goes_like_samples(lwp_all, sampling="hour", lwp_range_gm2=LWP_RANGE_GM2)
    rain_times = np.unique(rain.index[(rain > 0).to_numpy()].values.astype("datetime64[ns]"))  # rate > 0

    # --- low cloud fraction and cloud bases: Pier ceilometer, the notebook's definitions -----------------
    ceil = load_product("cbh_ceil_M1")
    lat, lon = float(ceil["lat"].values.ravel()[0]), float(ceil["lon"].values.ravel()[0])
    state = Q.ceilometer_low_cloud_state(ceil, max_cbh_m=LOW_CLOUD_MAX_M)  # 1 low cloud / 0 / NaN, per 16 s
    cbh, _ = Q.qc0_values(ceil, "first_cbh")  # m above the instrument, QC = 0
    del ceil
    cf = Q.low_cloud_fraction(state, window_min=WINDOW_MIN, min_coverage=CF_MIN_COVERAGE)["low_cloud_fraction"]
    cbh_low = Q.to_series(cbh).where(state == Q.LOW_CLOUD)  # the bases counted as "low cloud" in the fraction
    cbh_hourly = cbh_low.groupby(cbh_low.index.floor(hour)).mean()

    # --- cloud top: KAZR-ARSCL, top of the lowest hydrometeor layer whose base is <= LOW_CLOUD_MAX_M -------
    # (layer 0 = lowest; empty slots hold -1; heights in m above ground; no QC field is distributed).
    # Only the seasons' days are read (the file holds 7.9 M samples x 10 layers).
    arscl = load_product("cloud_arscl_M1")
    layer_dim = [d for d in arscl["cloud_layer_top_height"].dims if d != "time"][0]
    tops = []
    for season in seasons:
        for first, stop in season_slices(season):
            sub = arscl.sel(time=slice(first, stop - pd.Timedelta(seconds=1)))
            base0 = sub["cloud_layer_base_height"].isel({layer_dim: 0}).values
            top0 = sub["cloud_layer_top_height"].isel({layer_dim: 0}).values
            ok = (np.isfinite(base0) & (base0 >= 0) & (base0 <= LOW_CLOUD_MAX_M) & np.isfinite(top0)
                  & (top0 > base0))
            tops.append(pd.Series(np.where(ok, top0, np.nan), index=pd.DatetimeIndex(sub["time"].values)))
    del arscl
    top = pd.concat(tops).sort_index()
    top_hourly = top.groupby(top.index.floor(hour)).mean()

    cosz = (Q.to_series(load_product("radflux_M1")["cosine_zenith"]) if sza_source == "radflux" else None)
    if sza_source not in ("noaa", "radflux"):
        raise ValueError(f"sza source must be 'noaa' or 'radflux', not {sza_source!r}")

    tables = {}
    for season in seasons:
        lwp = lwp_all[S.season_mask(lwp_all.index, season)]
        to_rain = hours_to_rain(lwp.index, rain_times)  # hours from each kept sample to rain
        # --- hourly values: the MEAN of the hour's kept samples (the table's downsampling) ----------------
        key = lwp.index.floor(hour)
        out = pd.DataFrame({"lwp_gm2": lwp.groupby(key).mean(), "n_samples": lwp.groupby(key).size()})
        for col in ("closest_h", "prev_h", "next_h"):
            out[col] = to_rain[col].groupby(key).mean()
        out["low_cloud_fraction"] = cf.reindex(out.index)
        out["cbh_m"] = cbh_hourly.reindex(out.index)
        out["cloud_top_m"] = top_hourly.reindex(out.index)
        # --- daytime: solar zenith angle at mid-hour ------------------------------------------------------
        mid = out.index + hour / 2
        out["sza_deg"] = (Q.radflux_zenith_deg(cosz, mid) if cosz is not None else Q.solar_zenith_deg(mid, lat, lon))
        out["daytime"] = out["sza_deg"] <= SZA_MAX_DEG  # NaN (RADFLUX gap) = not daytime
        tables[season] = out[out["low_cloud_fraction"].notna()]  # hours without a cloud fraction never qualify
    return tables


# ---------------------------------------------------------------------------
# statistics vs the low-cloud-fraction threshold
# ---------------------------------------------------------------------------
def by_threshold(hours: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """Over the hours with low cloud fraction >= each threshold: the median, 25 % and 75 % of every hourly
    value (the curves), plus the mean and std of the hourly LWP (the table's statistic, for the check)."""
    rows = []
    for x in thresholds:
        h = hours[hours["low_cloud_fraction"] >= x - 1e-9]  # tolerance: 1.0 must keep fractions of exactly 1
        row = {"threshold": x, "n_hours": len(h),
               "lwp_gm2_mean": h["lwp_gm2"].mean(), "lwp_gm2_std": h["lwp_gm2"].std(ddof=1)}
        for col in HOURLY_COLUMNS:
            v = h[col].dropna()
            row[f"{col}_p50"] = v.median() if len(v) else np.nan
            row[f"{col}_p25"] = v.quantile(0.25) if len(v) else np.nan
            row[f"{col}_p75"] = v.quantile(0.75) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("threshold")


# ---------------------------------------------------------------------------
# panels (shared by both figures)
# ---------------------------------------------------------------------------
def overcast_line(ax, label: bool = False, n_hours: Optional[int] = None) -> None:
    """Dashed vertical line at the table's overcast threshold; its label can give the number of overcast hours
    (low cloud fraction >= OVERCAST_CF exactly, i.e. the hours behind the table's overcast value)."""
    ax.axvline(OVERCAST_CF, color=INK_3, lw=1, ls="--", zorder=1)
    if label:
        text = f"overcast (x ≥ {OVERCAST_CF:g})" + (f"\nn = {n_hours} hours" if n_hours is not None else "")
        ax.text(OVERCAST_CF - 0.01, 0.98, text, transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8,
                color=INK_2, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def n_overcast_hours(hours: pd.DataFrame) -> int:
    """Hours with low cloud fraction >= OVERCAST_CF (the hours behind the table's overcast value)."""
    return int((hours["low_cloud_fraction"] >= OVERCAST_CF - 1e-9).sum())


def draw_lwp(ax, curve: pd.DataFrame, title: str, legend: bool = True, n_overcast: Optional[int] = None) -> None:
    """Median and 25-75 % of the hourly LWP vs the threshold, with the number of hours at a few thresholds.
    n_overcast: hours with low cloud fraction >= OVERCAST_CF, shown in the label of the dashed overcast line."""
    x = curve.index.to_numpy()
    ax.fill_between(x, curve["lwp_gm2_p25"], curve["lwp_gm2_p75"], color=LWP_COLOR, alpha=0.15, lw=0,
                    label="25-75 % of the hourly LWP")
    ax.plot(x, curve["lwp_gm2_p50"], color=LWP_COLOR, marker="o", label="median LWP")
    overcast_line(ax, label=True, n_hours=n_overcast)
    # number of hours behind three markers: the first threshold, the middle one and the last (x = 1.0). Each label
    # names its threshold and a thin black leader line points to its marker. All three sit ABOVE the curve, which
    # only rises with x, so no label covers it. The overcast count goes in the dashed line's own label (above).
    # (label offset in points from the marker: dx, dy; text alignment: ha, va)
    picks = ((x[0], 14, 24, "left", "bottom"),
             (x[np.argmin(np.abs(x - 0.5))], 0, 24, "center", "bottom"),
             (x[-1], -80, 30, "right", "bottom"))
    for xi, dx, dy, ha, va in picks:
        ax.annotate(f"n = {int(curve.loc[xi, 'n_hours'])} hours (x = {xi:.2f})", (xi, curve.loc[xi, "lwp_gm2_p50"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=3))
    ax.set_ylabel("LWP (g m⁻²)")
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="upper left")


def box_data(hours: pd.DataFrame):
    """Hourly LWP of the hours with low cloud fraction >= each BOX_THRESHOLDS value."""
    return [hours.loc[hours["low_cloud_fraction"] >= b - 1e-9, "lwp_gm2"].to_numpy() for b in BOX_THRESHOLDS]


def draw_box(ax, hours: pd.DataFrame, title: str, top_y: float) -> None:
    """Box plots of the hourly LWP at BOX_THRESHOLDS; numbers above the boxes = hours. top_y sets the y range."""
    data = box_data(hours)
    keep = [i for i, d in enumerate(data) if d.size]
    bp = ax.boxplot([data[i] for i in keep], positions=BOX_THRESHOLDS[keep], widths=0.055, whis=1.5,
                    patch_artist=True, manage_ticks=False,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK_2, lw=1),
                    capprops=dict(color=INK_2, lw=1), boxprops=dict(edgecolor=INK_2, lw=1),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=INK_3, markeredgecolor="none"))
    for patch in bp["boxes"]:
        patch.set_facecolor(LWP_COLOR)
        patch.set_alpha(0.35)
    for i in keep:
        ax.text(BOX_THRESHOLDS[i], top_y * 1.04, f"{data[i].size} h", ha="center", va="bottom", fontsize=7.5,
                color=INK_2)
    overcast_line(ax)
    ax.set_ylim(0, top_y * 1.14)
    ax.set_ylabel("LWP (g m⁻²)")
    ax.set_title(title, loc="left")


def draw_clouds(ax, curve: pd.DataFrame, title: str, legend: bool = True) -> None:
    """Median cloud base (ceilometer) and cloud top (KAZR-ARSCL) vs the threshold."""
    x = curve.index.to_numpy()
    ax.fill_between(x, curve["cbh_m_p50"], curve["cloud_top_m_p50"], color=GRID, lw=0, zorder=1)
    ax.plot(x, curve["cloud_top_m_p50"], color=INK, ls="--", marker="o", label="cloud top (KAZR-ARSCL)")
    ax.plot(x, curve["cbh_m_p50"], color=INK, marker="o", label="cloud base (ceilometer)")
    overcast_line(ax)
    ax.set_ylabel("Height above ground (m)")
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="best")  # the empty corner differs between seasons (low summer decks, high winter ones)


def rain_axis_top(curves: Sequence[pd.DataFrame], prev_next: bool = SHOW_PREV_NEXT_RAIN) -> float:
    """Upper limit (h) of the time-to-rain axis: 4 x the largest value drawn (room for the legend), >= 60 h."""
    cols = ("closest_h_p75", "prev_h_p50", "next_h_p50") if prev_next else ("closest_h_p75",)
    vals = [c[col].replace([np.inf, -np.inf], np.nan).max() for c in curves for col in cols]
    return max(60.0, 4.0 * np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 60.0


def draw_rain(ax, curve: pd.DataFrame, title: str, rain_exclude_min: float, legend: bool = True,
              y_top: Optional[float] = None, prev_next: bool = SHOW_PREV_NEXT_RAIN) -> None:
    """Median and 25-75 % of the hourly mean time to the closest rain; with prev_next also the medians of the
    time since the previous and until the next rain. y_top: upper axis limit (h); default from this curve."""
    x = curve.index.to_numpy()
    ax.fill_between(x, curve["closest_h_p25"], curve["closest_h_p75"], color=INK_3, alpha=0.18, lw=0,
                    label="closest rain: 25-75 % of hours")
    ax.plot(x, curve["closest_h_p50"], color=INK, marker="o", label="closest rain (before or after): median")
    if prev_next:
        ax.plot(x, curve["prev_h_p50"], color=INK_2, lw=1, ls=":", label="since the previous rain: median")
        ax.plot(x, curve["next_h_p50"], color=INK_2, lw=1, ls="-.", label="until the next rain: median")
    ax.axhline(rain_exclude_min / 60, color=INK_3, lw=1, ls="--")
    ax.text(0.01, rain_exclude_min / 60, f" rain screen: ± {rain_exclude_min:g} min", va="bottom", fontsize=8,
            color=INK_2)
    overcast_line(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.4, y_top if y_top is not None else rain_axis_top([curve], prev_next))  # headroom for the legend
    plain_log_axis(ax, which="y")
    ax.set_ylabel("Time to rain (h)")
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="upper left", fontsize=8, ncol=2)


def filters_text(seasons: Sequence[str], time_of_day: str, rain_gauges: str, rain_exclude_min: float,
                 sza_source: str, box_panel: bool) -> str:
    """The filters, for the text box."""
    when = (f"daytime: solar zenith angle\n  <= {SZA_MAX_DEG:g}° at mid-hour ({'NOAA equations' if sza_source == 'noaa' else 'RADFLUX'})"
            if time_of_day == "daytime" else "all time: any hour of the day")
    gauges = "Pier Parsivel only" if rain_gauges == "pier_parsivel" else "any gauge (Pier Parsivel,\n  Soledad Parsivel, Pier VDIS)"
    lo, hi = LWP_RANGE_GM2
    season_lines = []
    for season in seasons:  # e.g. "Winter: 2023-02-15..2023-03-31\n  + 2024-01-01..2024-02-14"
        spans = [f"{a}..{b}" for a, b in S.SEASONS[season]]
        season_lines.append(f"{season}: {spans[0]}" + "".join(f"\n  + {sp}" for sp in spans[1:]))
    lines = [
        "Filters (as the seasonal table,",
        "Section 3c, 'hour' sampling)",
        "",
        *season_lines,
        "  (UTC)",
        "",
        "LWP: MWRRET v2 phys_lwp",
        "  (3-channel MWR, ~1.3 s)",
        "  qc_phys_lwp = 0, phys_qc_flag = 0",
        f"  {lo:g} <= LWP <= {hi:g} g m-2 per sample",
        "",
        "Rain screen: no rain within",
        f"  ± {rain_exclude_min:g} min; rain = any",
        "  1-min rate > 0 mm/h",
        f"  gauges: {gauges}",
        "",
        "Hourly values: data that passed",
        "  these filters were downsampled",
        "  by taking the mean over each",
        "  UTC hour (as the table): LWP,",
        "  cloud base, cloud top and time",
        "  to rain. Curves: median and",
        "  25-75 % of those hourly values;",
        "  n = hours",
        "",
        "Low cloud fraction of the hour:",
        "  share of the Pier ceilometer's",
        "  16-s samples with a base",
        f"  <= {LOW_CLOUD_MAX_M:g} m (status 1-3,",
        "  qc_first_cbh = 0); needs data",
        f"  for >= {100 * CF_MIN_COVERAGE:g} % of the hour",
        "Threshold x: hours with",
        f"  fraction >= x (x = {OVERCAST_CF:g}:",
        "  the table's 'overcast')",
        "",
        when,
        "",
        "Cloud base: ceilometer, the",
        "  low-cloud bases above",
        "Cloud top: KAZR-ARSCL, top of",
        "  the lowest layer with base",
        f"  <= {LOW_CLOUD_MAX_M:g} m",
        "Time to rain: from each kept",
        "  LWP sample to the closest rain",
    ]
    if box_panel:
        lines += ["", "Box plots: box 25-75 %, line", "  median, whiskers 1.5 x box",
                  "  height (Tukey), dots beyond"]
    return "\n".join(lines)


def draw_filter_box(fig, spec, text: str) -> None:
    """The filter text in its own (axis-less) cell of the figure grid."""
    tax = fig.add_subplot(spec)
    tax.axis("off")
    tax.text(0.04, 1.0, text, va="top", ha="left", fontsize=8.6, color=INK, family="DejaVu Sans", linespacing=1.25,
             bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=GRID, lw=0.8), transform=tax.transAxes)


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
X_LABEL = "Low cloud fraction threshold x (hours whose ceilometer low cloud fraction ≥ x)"


def plot_season(season: str, hours: pd.DataFrame, curve: pd.DataFrame, time_of_day: str, rain_gauges: str,
                rain_exclude_min: float, sza_source: str, path: Path, box_panel: bool = SHOW_BOX_PANEL,
                prev_next: bool = SHOW_PREV_NEXT_RAIN) -> None:
    """One season: LWP, (box plots,) cloud boundaries, time to rain, stacked; filter box on the right."""
    use_style()
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    kinds = ["lwp"] + (["box"] if box_panel else []) + ["clouds", "rain"]
    n = len(kinds)
    fig = plt.figure(figsize=(12.5, 3.6 * n + 0.5))
    gs = fig.add_gridspec(n, 2, width_ratios=[3.3, 1.0], hspace=0.3, wspace=0.04, top=1 - 0.18 / n, bottom=0.2 / n)
    axes = {kind: fig.add_subplot(gs[i, 0]) for i, kind in enumerate(kinds)}
    for kind in kinds[1:]:
        axes[kind].sharex(axes["lwp"])
    letter = {kind: "abcd"[i] for i, kind in enumerate(kinds)}
    gauge_word = "the Pier Parsivel" if rain_gauges == "pier_parsivel" else "any gauge"

    draw_lwp(axes["lwp"], curve, f"{letter['lwp']}. Median LWP, hours with low cloud fraction ≥ x",
             n_overcast=n_overcast_hours(hours))
    if box_panel:
        top_y = max(np.nanmax(d) for d in box_data(hours) if d.size)
        draw_box(axes["box"], hours, f"{letter['box']}. LWP for x = 0.1, …, 1.0 (numbers = hours)",
                 top_y)
    draw_clouds(axes["clouds"], curve, f"{letter['clouds']}. Median cloud base and cloud top, hours with low cloud "
                "fraction ≥ x")
    draw_rain(axes["rain"], curve, f"{letter['rain']}. Time from the kept LWP samples to rain at {gauge_word} "
              "vs x", rain_exclude_min, prev_next=prev_next)
    axes["rain"].set_xlabel(X_LABEL)
    axes["rain"].set_xlim(-0.02, 1.04)
    for kind in kinds[:-1]:
        axes[kind].tick_params(labelbottom=False)
    draw_filter_box(fig, gs[:, 1], filters_text([season], time_of_day, rain_gauges, rain_exclude_min, sza_source,
                                                box_panel))
    fig.suptitle(f"EPCAPE {season.lower()}, GOES-like MWR LWP vs hourly low cloud fraction ({time_of_day}; "
                 f"{len(hours)} hours with LWP and a cloud fraction)", x=0.02, ha="left", y=0.995,
                 fontsize=12.5, fontweight="semibold", color=INK)
    save(fig, path)


def plot_compare(seasons: Sequence[str], hours: Dict[str, pd.DataFrame], curves: Dict[str, pd.DataFrame],
                 time_of_day: str, rain_gauges: str, rain_exclude_min: float, sza_source: str, path: Path,
                 box_panel: bool = SHOW_BOX_PANEL, prev_next: bool = SHOW_PREV_NEXT_RAIN) -> None:
    """Seasons side by side (one column each, same y axis in every row); filter box in the right column."""
    use_style()
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    kinds = ["lwp"] + (["box"] if box_panel else []) + ["clouds", "rain"]
    n, m = len(kinds), len(seasons)
    fig = plt.figure(figsize=(7.4 * m + 3.6, 3.6 * n + 0.6))
    gs = fig.add_gridspec(n, m + 1, width_ratios=[3.0] * m + [1.15], hspace=0.32, wspace=0.08,
                          top=1 - 0.2 / n, bottom=0.2 / n)
    axes = {}
    for j, season in enumerate(seasons):
        for i, kind in enumerate(kinds):
            share_x = axes.get((seasons[0], "lwp"))
            share_y = axes.get((seasons[0], kind))  # same y axis across the seasons in each row
            axes[(season, kind)] = fig.add_subplot(gs[i, j], sharex=share_x, sharey=share_y)
    gauge_word = "the Pier Parsivel" if rain_gauges == "pier_parsivel" else "any gauge"
    top_y = (max(np.nanmax(d) for s in seasons for d in box_data(hours[s]) if d.size) if box_panel else None)
    letters = iter("abcdefghijkl")  # column by column: the first season a, b, c, then the next
    for j, season in enumerate(seasons):
        first_col = j == 0
        label = SEASON_LABELS.get(season, season)
        draw_lwp(axes[(season, "lwp")], curves[season],
                 f"{next(letters)}. {label}: median LWP", legend=first_col,
                 n_overcast=n_overcast_hours(hours[season]))
        if box_panel:
            draw_box(axes[(season, "box")], hours[season], f"{next(letters)}. {season}: LWP for x = 0.1, …, 1.0 "
                     "(numbers = hours)", top_y)
        draw_clouds(axes[(season, "clouds")], curves[season],
                    f"{next(letters)}. {season}: median cloud base and top", legend=first_col)
        draw_rain(axes[(season, "rain")], curves[season],
                  f"{next(letters)}. {season}: time to rain at {gauge_word}", rain_exclude_min,
                  legend=first_col, y_top=rain_axis_top([curves[s] for s in seasons], prev_next),
                  prev_next=prev_next)
        axes[(season, "rain")].set_xlabel(X_LABEL if m == 1 else "Low cloud fraction threshold x (hours with fraction ≥ x)")
        axes[(season, "rain")].set_xlim(-0.02, 1.04)
        for kind in kinds[:-1]:
            axes[(season, kind)].tick_params(labelbottom=False)
        if not first_col:  # the shared y axis is labelled once, on the left
            for kind in kinds:
                axes[(season, kind)].tick_params(labelleft=False)
                axes[(season, kind)].set_ylabel("")
    draw_filter_box(fig, gs[:, m], filters_text(seasons, time_of_day, rain_gauges, rain_exclude_min, sza_source,
                                                box_panel))
    counts = "; ".join(f"{s.lower()} {len(hours[s])} h" for s in seasons)
    fig.suptitle(f"EPCAPE {' vs '.join(s.lower() for s in seasons)}, GOES-like MWR LWP vs hourly low cloud fraction "
                 f"({time_of_day}; hours with LWP and a cloud fraction: {counts})", x=0.02, ha="left", y=0.995,
                 fontsize=12.5, fontweight="semibold", color=INK)
    save(fig, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    # The low-cloud limit is a module setting read by hourly_tables and filters_text at call time; --cbh-max-m
    # sets it for this run.
    global LOW_CLOUD_MAX_M
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--rain-gauges", choices=["pier_parsivel", "any"], default=RAIN_GAUGES,
                   help="gauges that define rain for the screen and the time-to-rain panel (default: %(default)s)")
    p.add_argument("--rain-exclude-min", type=float, default=RAIN_EXCLUDE_MIN,
                   help="drop MWR samples within ± this many minutes of rain (default: %(default)s)")
    p.add_argument("--sza-source", choices=["noaa", "radflux"], default=SZA_SOURCE,
                   help="solar zenith angle for the daytime cut (default: %(default)s)")
    p.add_argument("--box-panel", action="store_true", default=SHOW_BOX_PANEL,
                   help="add the LWP box-plot panel (default: off)")
    p.add_argument("--prev-next-rain", action="store_true", default=SHOW_PREV_NEXT_RAIN,
                   help="also plot the time since the previous and until the next rain (default: off)")
    p.add_argument("--cbh-max-m", type=float, default=LOW_CLOUD_MAX_M,
                   help="low-cloud limit: cloud bases at or below this height (m) count as low cloud, for the cloud "
                        "fraction, the cloud base and the ARSCL layer behind the cloud top (default: %(default)s)")
    p.add_argument("--out", type=Path, default=None, help="output folder")
    args = p.parse_args(argv)
    LOW_CLOUD_MAX_M = args.cbh_max_m

    out_dir = args.out or (Path(__file__).resolve().parent / "winter_LWP_investigation")  # next to this script
    out_dir.mkdir(parents=True, exist_ok=True)
    # file-name tag: the rain gauges and the screen (and the low-cloud limit when it is not the default 3500 m),
    # so each choice keeps its own files
    tag = f"{'pier_parsivel' if args.rain_gauges == 'pier_parsivel' else 'any_gauge'}_{args.rain_exclude_min:g}min"
    if LOW_CLOUD_MAX_M != DEFAULT_LOW_CLOUD_MAX_M:
        tag += f"_cbh{LOW_CLOUD_MAX_M:g}m"
    seasons = list(dict.fromkeys([SEASON, *COMPARE_SEASONS]))  # every season needed, each once

    tables = hourly_tables(seasons, args.rain_gauges, args.rain_exclude_min, args.sza_source)
    for season, hours in tables.items():
        hours.to_csv(out_dir / f"{season.lower()}_LWP_hourly_{tag}.csv", index_label="hour_start_utc")
        print(f"{season}: {len(hours)} hours with kept LWP samples and a cloud fraction "
              f"({int(hours['daytime'].sum())} daytime); rain gauges {args.rain_gauges}, ± {args.rain_exclude_min:g} min")

    for time_of_day in ("daytime", "all time"):
        short = "daytime" if time_of_day == "daytime" else "alltime"
        hours_tod, curves = {}, {}
        for season, hours in tables.items():
            h = hours[hours["daytime"]] if time_of_day == "daytime" else hours
            hours_tod[season], curves[season] = h, by_threshold(h, CF_THRESHOLDS)
            curves[season].to_csv(out_dir / f"{season.lower()}_LWP_by_threshold_{short}_{tag}.csv")
            ov = by_threshold(h, np.array([OVERCAST_CF])).iloc[0]  # exactly the table's overcast cut
            print(f"  {season}, {time_of_day}: {len(h)} hours; at x = {OVERCAST_CF:g}: {int(ov['n_hours'])} hours, LWP "
                  f"median {ov['lwp_gm2_p50']:.0f} g m-2 (mean ± std {ov['lwp_gm2_mean']:.0f} ± {ov['lwp_gm2_std']:.0f}, "
                  f"the table's statistic), base {ov['cbh_m_p50']:.0f} m, top {ov['cloud_top_m_p50']:.0f} m, "
                  f"closest rain {ov['closest_h_p50']:.1f} h")
        path = out_dir / f"{SEASON.lower()}_LWP_vs_low_cloud_fraction_{short}_{tag}"
        plot_season(SEASON, hours_tod[SEASON], curves[SEASON], time_of_day, args.rain_gauges, args.rain_exclude_min,
                    args.sza_source, path, box_panel=args.box_panel, prev_next=args.prev_next_rain)
        cpath = out_dir / f"{'_vs_'.join(s.lower() for s in COMPARE_SEASONS)}_LWP_vs_low_cloud_fraction_{short}_{tag}"
        plot_compare(COMPARE_SEASONS, hours_tod, curves, time_of_day, args.rain_gauges, args.rain_exclude_min,
                     args.sza_source, cpath, box_panel=args.box_panel, prev_next=args.prev_next_rain)
        print(f"  -> {path.name}.png, {cpath.name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
