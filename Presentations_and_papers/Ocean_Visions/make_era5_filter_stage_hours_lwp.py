#!/usr/bin/env python3
"""What do the overcast and no-precipitation filters do to hours and to LWP?

WHAT IT MAKES
=============
Three figures for October, November and December, for pack ice, open ocean, land
and the UTQ grid cell, each drawn with area weighting and with equal weighting:

    1. Hours per month   stacked bars, median over the seasons
    2. Hours per day     stacked bars, median over the days, the statistic in the
                         median line of Genie's daily box plots (her Figure 5)
    3. LWP               three bars per class: median over the seasons of the mean
                         liquid water path in the hours that pass each stage

In a stacked bar, the whole bar is the hours at the first stage and the segments
show the hours each filter removes.

THE THREE FILTER STAGES
=======================
All three require a liquid-containing column: cloud water above the 0.01 g m-2
floors and IWP/CWP < 0.90. They then tighten the cloud cover and precipitation
tests one at a time:

    any_cloud          tcc > 0, with no precipitation test
    overcast           tcc >= 0.95
    overcast_noprecip  tcc >= 0.95 and tp < 0.05 mm hr-1  (the project's tables)

tcc > 0 is needed as its own test because ERA5 does not guarantee it: from
October to December, 0.107% of liquid-containing cell-hours have tcc = 0 (mean
LWP 0.16 g m-2). Each stage is a subset of the one before. The order matters for
the labels. "Removed by the no-precipitation filter" means overcast hours that
precipitate. A precipitating hour that is not overcast is already gone at the
first step, so it counts as removed by the overcast filter.

HOW THE VALUES ARE DEFINED
==========================
For one class c and hour t, with weight w (1, or cos(lat) for area weighting):

    N(t)  = sum of w over the valid cells of class c
    n(t)  = sum of w over those cells that pass the stage
    L(t)  = sum of w * LWP over those same cells             [g m-2]

Over any period (one UTC day, or one month of one season):

    hours per cell  H = sum_t n(t) / N(t)                    [h]
    mean LWP          = [ sum_t L(t) / N(t) ] / H            [g m-2]

H is the same per_hour average as make_era5_class_per_cell_csvs.py. Hours with no
cells of the class add zero to H, so open ocean in December is small partly
because it often does not exist. Mean LWP uses the same hourly normalization, so
each hour counts in proportion to the share of the class that passes the stage.

Hours per month are summarized by the median of the seasons' monthly H. Hours per
day are summarized by the median of H over every day of that month, pooled across
the seasons. A season with H = 0 counts as zero for hours. It has no defined LWP,
so it is left out of the LWP median, and the figure says so.

WHY MEDIANS CAN STILL BE STACKED
================================
Medians do not add, but every day and season has any_cloud >= overcast >=
overcast_noprecip, so the medians of the three stages keep that order. A segment
is therefore a DIFFERENCE BETWEEN TWO MEDIANS, not the median of the hours that
filter removes. The bar tops and the purple segments are true medians.

WHY THE LWP BARS ARE NOT STACKED
================================
Mean LWP can rise or fall as a filter removes hours, so a stack of LWP values
would have no meaning.

COMPARING WITH GENIE'S FIGURES
==============================
The figures use all the seasons. Genie's Figure 5 leaves out the months with
incomplete ARM data, which here means October-December 2020. The tables also hold
that version of each median, in the column median_excl_arm_missing. With it,
the equal-weighted daily medians agree with her Figure 5 to about 0.1 h.

WHY THIS SCRIPT MAKES ITS OWN PASS
==================================
count_cell_hours does not record LWP, so this script reads the raw fields and
evaluates all three stages on the same hours. It calls the project's
classify_cells and fraction_phase_masks and takes its thresholds from SETTINGS,
so the definitions are not copied. check_against_csvs compares the last stage with
the monthly and daily tables, and every run tests that the stages are nested.

WHAT "SUPERCOOLED" MEANS HERE
=============================
As in the rest of the Ocean Visions tables, the phase test uses tclw and tciw and
contains no temperature test. make_era5_supercooled_check.py examines how much of
that liquid is below 0 deg C.

USAGE
=====
    python make_era5_filter_stage_hours_lwp.py                     # both weightings
    python make_era5_filter_stage_hours_lwp.py --cell-weighting uniform
    python make_era5_filter_stage_hours_lwp.py --plot-only         # no raw data read

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

# First, before numpy: on Casper it sets thread limits and the cache location,
# and it puts the analysis modules on sys.path.
import ov_config as ov

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from make_era5_hour_count_csvs import (  # noqa: E402
    I_UTQ,
    OUT_PRODUCT,
    SETTINGS,
    SLOT_CLASS,
    SLOT_NAMES,
    build_args,
    git_describe,
    season_label,
)

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from cloud_classification import fraction_phase_masks  # noqa: E402
from plot_surface_class_timeseries import (  # noqa: E402
    season_layout,
    select_seasons,
    site_cell_mask,
    site_in_grid,
)
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    CLASS_ORDER,
    UNCLASSIFIED,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

# ----------------------------------------------------------------------------
# What is computed and drawn
# ----------------------------------------------------------------------------
MONTHS = {10: "Oct", 11: "Nov", 12: "Dec"}       # calendar month -> label
STAGES = ("any_cloud", "overcast", "overcast_noprecip")
STAGE_LABELS = {
    "any_cloud": "Cloud present (tcc > 0)",
    "overcast": "Overcast (tcc ≥ 0.95)",
    "overcast_noprecip": "Overcast and no precipitation (tp < 0.05 mm/hr)",
}
# Classes in the figures, in the order requested, with their tick labels.
FIGURE_CLASSES = {"PackIce": "Pack ice", "OpenOcean": "Open ocean", "Land": "Land", "UTQ": "UTQ"}

# One colour per filter stage, on every bar and in both figures. The purple and
# teal are the pair in make_era5_precip_filter_sensitivity.py: purple = passes
# both filters, teal = passes overcast but precipitates. They and the grey are
# well separated in lightness, so they stay distinct under colour-vision
# deficiency and in greyscale.
COLOR_BOTH = "#5e3c99"
COLOR_OVERCAST = "#2ca58d"
COLOR_ANY_CLOUD = "#cfcfcf"
STAGE_COLORS = {
    "any_cloud": COLOR_ANY_CLOUD,
    "overcast": COLOR_OVERCAST,
    "overcast_noprecip": COLOR_BOTH,
}
INK_PRIMARY = "#222222"    # titles, axis labels and values
INK_MUTED = "#666666"      # secondary annotations
GRID_COLOR = "#e4e4e4"     # recessive gridlines
WEIGHT_NOTES = {"area": "Area weighted (cos latitude).", "uniform": "Equal weight per grid cell."}
MIN_SEGMENT_LABEL_H = 25.0        # h; thinner monthly stack segments carry no number
MIN_SEGMENT_LABEL_H_DAY = 0.8     # h; the same for the daily figure

# Genie's Figure 5 leaves out months with incomplete ARM data. In these three
# calendar months that is 2020/21 only (her caption: October-December 2020).
ARM_MISSING_SEASON = "2020/21"
# Columns of the monthly tables that are not season values.
NON_SEASON_COLUMNS = {"month", "class", "stage", "mean_of_seasons", "median_of_seasons",
                      "median_excl_arm_missing", "n_seasons"}


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def accumulate(args: argparse.Namespace) -> dict:
    """Hourly n(t), L(t), N(t) per slot and stage for October-December.

    The setup and classification follow count_cell_hours, so the population is
    the same one the hour-count tables use.

    Returns arrays ``n`` and ``lwp`` shaped (hour, slot, stage), ``size`` and
    ``present`` shaped (hour, slot), and each hour's season and month.
    """
    region_dir = resolve_region_dir(args)
    ds = load_seb_data(args.region, None, None, region_dir.parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(
            args.region, lwph.resolve_data_root(args.storage, args.data_root), args.mask_grid
        ),
        ds,
    )

    # Season bookkeeping from the time axis only, as in the hour-count tables.
    layout = season_layout(ds, args)
    keep_idx, used, _label = select_seasons(layout, args)
    s_idx, in_window = layout["s_idx"], layout["in_window"]
    wanted = np.zeros(len(layout["seasons"]), dtype=bool)
    wanted[keep_idx] = True
    times = np.asarray(ds["valid_time"].values)
    time_index = pd.DatetimeIndex(times)
    # Only the three months in the figures are read.
    use_step = (
        in_window
        & (s_idx >= 0)
        & wanted[np.clip(s_idx, 0, None)]
        & np.isin(time_index.month, list(MONTHS))
    )
    season_years = np.asarray(layout["seasons"])
    hour_season = season_years[s_idx[use_step]]          # (hour,) season start year
    hour_month = time_index.month.values[use_step]       # (hour,) calendar month
    hour_date = times[use_step].astype("datetime64[D]")  # (hour,) UTC calendar day
    row_of_step = np.cumsum(use_step) - 1                # time step -> output row

    has_site = site_in_grid(ds)
    site_mask = site_cell_mask(ds)[0] if has_site else None
    n_hour, n_slot, n_stage = int(use_step.sum()), len(SLOT_NAMES), len(STAGES)
    n_sum = np.zeros((n_hour, n_slot, n_stage))          # sum of w over passing cells
    lwp_sum = np.zeros((n_hour, n_slot, n_stage))        # sum of w * LWP, g m-2
    size = np.zeros((n_hour, n_slot))                    # N(t), sum of w over valid cells
    count = np.zeros((n_hour, n_slot), dtype=np.int32)   # valid cells, unweighted

    # cos(lat) when area weighting, else 1. Shape (lat, lon).
    if args.cell_weighting == "area":
        weight = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])
    else:
        weight = np.ones((ds.sizes["latitude"], ds.sizes["longitude"]))

    # Output slot of each class code; unclassified cells go to a scratch slot.
    slot_of_code = np.empty(len(CLASS_ORDER), dtype=np.intp)
    for slot_name, cls in SLOT_CLASS.items():
        slot_of_code[CLASS_CODES[cls]] = SLOT_NAMES.index(slot_name)
    scratch = n_slot

    read_vars = ["tcc", args.liquid_var, "tciw", "tp", "siconc"]
    print(f"  Streaming {n_hour:,} hours ...", flush=True)
    for i0, block in iter_time_blocks(ds, read_vars, args.block_hours, keep_mask=use_step):
        sl = slice(i0, i0 + block.sizes["valid_time"])
        keep = use_step[sl]
        if not keep.any():
            continue
        rows = row_of_step[sl][keep]                     # (t,) output rows
        n_t = rows.size

        classes = classify_cells(
            lsm,
            block["siconc"].values,
            args.lsm_tol,
            args.open_ocean_max_siconc,
            args.sea_ice_min_siconc,
            args.land_max_siconc,
        )[keep]                                          # (t, lat, lon)
        slot = np.where(
            classes == UNCLASSIFIED, scratch, slot_of_code[np.clip(classes, 0, None)]
        )

        tcc = block["tcc"].values[keep]                          # fraction, 0-1
        lwp_g_m2 = block[args.liquid_var].values[keep] * 1000.0  # kg m-2 -> g m-2
        iwp_g_m2 = block["tciw"].values[keep] * 1000.0           # kg m-2 -> g m-2
        rate_mm_hr = block["tp"].values[keep] * 1000.0           # m h-1 -> mm h-1

        valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)
        phase = fraction_phase_masks(
            lwp_g_m2,
            iwp_g_m2,
            args.liquid_fraction_min,
            args.ice_fraction_min,
            args.min_lwp,
            args.min_iwp,
        )
        liquid_containing = valid & (phase["cwp_g"] > 0.0) & (phase["liquid"] | phase["mixed"])
        # ERA5 can hold condensate above the floors with tcc = 0, so a cloud has to
        # be confirmed by the cloud cover as well as by the cloud water.
        cloud_present = valid & (tcc > 0.0)
        overcast = valid & (tcc >= args.min_cloud_fraction)
        raining = np.isfinite(rate_mm_hr) & (rate_mm_hr >= args.precip_rate_max)
        stage_masks = (
            liquid_containing & cloud_present,
            liquid_containing & overcast,
            liquid_containing & overcast & ~raining,
        )

        # Flat index (hour, slot), so one bincount fills a whole block.
        flat = (np.arange(n_t)[:, None, None] * (n_slot + 1) + slot).ravel()
        w_flat = np.broadcast_to(weight, valid.shape).ravel()    # (t * lat * lon,)
        lwp_flat = lwp_g_m2.ravel()
        n_bins = n_t * (n_slot + 1)

        def by_slot(mask: np.ndarray, values: np.ndarray | None) -> np.ndarray:
            """Sum ``values`` over ``mask`` per (t, slot), dropping the scratch slot."""
            m = mask.ravel()
            summed = np.bincount(
                flat[m], weights=None if values is None else values[m], minlength=n_bins
            )
            return summed.reshape(n_t, n_slot + 1)[:, :n_slot]

        count[rows] = by_slot(valid, None)
        size[rows] = by_slot(valid, w_flat)
        for k, mask in enumerate(stage_masks):
            n_sum[rows, :, k] = by_slot(mask, w_flat)
            lwp_sum[rows, :, k] = by_slot(mask, w_flat * np.nan_to_num(lwp_flat))

        # UTQ is one cell, recorded on top of the class that cell belongs to.
        if has_site:
            site_w = float(weight[site_mask][0])
            site_lwp = lwp_g_m2[:, site_mask][:, 0]              # (t,) g m-2
            site_valid = valid[:, site_mask][:, 0]               # (t,)
            count[rows, I_UTQ] = site_valid
            size[rows, I_UTQ] = site_valid * site_w
            for k, mask in enumerate(stage_masks):
                site_pass = mask[:, site_mask][:, 0]
                n_sum[rows, I_UTQ, k] = site_pass * site_w
                lwp_sum[rows, I_UTQ, k] = site_pass * site_w * np.nan_to_num(site_lwp)

    # Each stage must be a subset of the one before, or the stack is meaningless.
    if (n_sum[:, :, 1:] > n_sum[:, :, :-1] + 1e-9).any():
        raise AssertionError("a filter stage holds more cell-hours than the stage before it")

    return {
        "n": n_sum,
        "lwp": lwp_sum,
        "size": size,
        "present": count > 0,
        "hour_season": hour_season,
        "hour_month": hour_month,
        "hour_date": hour_date,
        "seasons": list(used),
    }


def build_tables(A: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly hours [h], monthly mean LWP [g m-2], and daily hours [h] with their summary.

    Season columns of the monthly tables hold that season's value. LWP is NaN when
    the stage has no hours in that month and season. Returns (hours_df, lwp_df,
    daily_df, daily_summary_df).
    """
    labels = [season_label(y) for y in A["seasons"]]
    # n(t)/N(t) and L(t)/N(t), and 0 where the class does not exist. (hour, slot, stage)
    denom = np.where(A["present"], A["size"], 1.0)[..., None]
    occupancy = np.where(A["present"][..., None], A["n"] / denom, 0.0)
    lwp_per_cell = np.where(A["present"][..., None], A["lwp"] / denom, 0.0)

    hour_rows, lwp_rows = [], []
    for month, month_name in MONTHS.items():
        for s, slot_name in enumerate(SLOT_NAMES):
            for k, stage in enumerate(STAGES):
                hours, lwp = {}, {}
                for year, label in zip(A["seasons"], labels):
                    sel = (A["hour_season"] == year) & (A["hour_month"] == month)
                    h = occupancy[sel, s, k].sum()
                    hours[label] = h
                    lwp[label] = lwp_per_cell[sel, s, k].sum() / h if h > 0 else np.nan
                key = {"month": month_name, "class": slot_name, "stage": stage}
                hour_rows.append({**key, **hours})
                lwp_rows.append({**key, **lwp})
    hours_df, lwp_df = pd.DataFrame(hour_rows), pd.DataFrame(lwp_rows)
    # Zero-hour seasons count for hours. NaN seasons are skipped for LWP.
    without_missing = [label for label in labels if label != ARM_MISSING_SEASON]
    for df in (hours_df, lwp_df):
        df["mean_of_seasons"] = df[labels].mean(axis=1, skipna=True)
        df["median_of_seasons"] = df[labels].median(axis=1, skipna=True)
        df["median_excl_arm_missing"] = df[without_missing].median(axis=1, skipna=True)
    lwp_df["n_seasons"] = lwp_df[labels].notna().sum(axis=1)

    # Daily hours per cell. A day with no cells of the class is a real zero, as in
    # the perDay tables and in Genie's box plots.
    dates, day_of_hour = np.unique(A["hour_date"], return_inverse=True)
    daily = np.zeros((dates.size, *occupancy.shape[1:]))       # (day, slot, stage) h
    np.add.at(daily, day_of_hour, occupancy)
    date_index = pd.DatetimeIndex(dates)
    daily_df = pd.DataFrame({"date": date_index.strftime("%Y-%m-%d")})
    daily_df["month"] = [MONTHS[m] for m in date_index.month]
    # Season start year: all three months fall in the year the season starts.
    daily_df["season"] = [season_label(y) for y in date_index.year]
    for s, slot_name in enumerate(SLOT_NAMES):
        for k, stage in enumerate(STAGES):
            daily_df[f"{slot_name}__{stage}"] = daily[:, s, k]

    summary_rows = []
    for month_name in MONTHS.values():
        in_month = daily_df["month"] == month_name
        kept_days = in_month & (daily_df["season"] != ARM_MISSING_SEASON)
        for slot_name in SLOT_NAMES:
            for stage in STAGES:
                column = f"{slot_name}__{stage}"
                summary_rows.append({
                    "month": month_name, "class": slot_name, "stage": stage,
                    "median_of_days": daily_df.loc[in_month, column].median(),
                    "mean_of_days": daily_df.loc[in_month, column].mean(),
                    "median_excl_arm_missing": daily_df.loc[kept_days, column].median(),
                    "n_days": int(in_month.sum()),
                })
    return hours_df, lwp_df, daily_df, pd.DataFrame(summary_rows)


def check_against_csvs(hours_df: pd.DataFrame, daily_df: pd.DataFrame, span: str,
                       hour_count_dir: Path, weighting: str) -> None:
    """The last stage must reproduce the existing monthly and daily hour tables."""
    suffix = ov.weight_suffix(weighting)
    folder = hour_count_dir / f"class_normalized{suffix}"
    stem = ov.weighted_names("ERA5_avgHoursPerCell_perMonth_", weighting)
    last = hours_df[hours_df["stage"] == "overcast_noprecip"].set_index(["month", "class"])
    worst, n_checked = 0.0, 0
    for slot_name in SLOT_NAMES:
        # UTQ is one cell, so its per-cell hours are its total hours in any weighting.
        if slot_name == "UTQ":
            path = hour_count_dir / f"ERA5_totalHours_perMonth_{span}_liquidContaining_UTQ.csv"
        else:
            path = folder / f"{stem}{span}_liquidContaining_{slot_name}.csv"
        if not path.exists():
            print(f"  !! no table to check against: {path.name}")
            continue
        table = pd.read_csv(path).set_index("month")
        for month_name in MONTHS.values():
            for season in table.columns:
                worst = max(
                    worst, abs(last.loc[(month_name, slot_name), season] - table.loc[month_name, season])
                )
                n_checked += 1

    # The daily tables use dates such as "1 Oct. 2014"; all three months abbreviate alike.
    day_stem = ov.weighted_names("ERA5_avgHoursPerCell_perDay_", weighting)
    mine_by_date = daily_df.assign(when=pd.to_datetime(daily_df["date"])).set_index("when")
    for slot_name in SLOT_NAMES:
        if slot_name == "UTQ":
            path = hour_count_dir / f"ERA5_totalHours_perDay_{span}_UTQ.csv"
        else:
            path = folder / f"{day_stem}{span}_{slot_name}.csv"
        if not path.exists():
            print(f"  !! no table to check against: {path.name}")
            continue
        table = pd.read_csv(path)
        table = table[table["date"].str.contains("Oct|Nov|Dec")]
        when = pd.to_datetime(table["date"], format="%d %b. %Y")
        mine = mine_by_date.loc[when, f"{slot_name}__overcast_noprecip"].values
        worst = max(worst, float(np.abs(mine - table["liquid_containing"].values).max()))
        n_checked += len(table)

    if n_checked:
        print(f"  check: last stage matches the monthly and daily tables, {n_checked} values, "
              f"worst {worst:.5f} h")
        if worst > 0.01:
            raise AssertionError(f"last stage differs from the existing tables by {worst:.3f} h")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def bar_positions() -> tuple[np.ndarray, list[float]]:
    """x of each class within each month, (n_month, n_class), and the month centres."""
    n_class = len(FIGURE_CLASSES)
    step = n_class + 1.0            # one empty slot between months
    x = np.array([[m * step + c for c in range(n_class)] for m in range(len(MONTHS))])
    return x, [float(row.mean()) for row in x]


def lookup(df: pd.DataFrame, month: str, cls: str, stage: str, column: str) -> float:
    """One value from a long table."""
    row = df[(df["month"] == month) & (df["class"] == cls) & (df["stage"] == stage)]
    return float(row[column].iloc[0])


def style_axes(ax, x: np.ndarray, centres: list[float]) -> None:
    """Class names on the bars, month names beneath them, and a quiet grid."""
    ax.set_xticks(x.ravel())
    ax.set_xticklabels([*FIGURE_CLASSES.values()] * len(MONTHS),
                       rotation=35, ha="right", fontsize=9.5, color=INK_PRIMARY)
    for centre, month_name in zip(centres, MONTHS.values()):
        ax.annotate(
            month_name, xy=(centre, 0), xycoords=("data", "axes fraction"), xytext=(0, -62),
            textcoords="offset points", ha="center", va="top",
            fontsize=13, fontweight="bold", color=INK_PRIMARY,
        )
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", colors=INK_MUTED)


def plot_stacked(df: pd.DataFrame, column: str, path: Path, title: str, ylabel: str,
                 weighting: str, y_max: float, y_ticks: np.ndarray, label_format: str,
                 min_label_h: float) -> Path:
    """A stacked hours figure; the whole bar is the any_cloud value of ``column``."""
    x, centres = bar_positions()
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    width = 0.82
    for m, month_name in enumerate(MONTHS.values()):
        for c, cls in enumerate(FIGURE_CLASSES):
            total = lookup(df, month_name, cls, "any_cloud", column)
            mid = lookup(df, month_name, cls, "overcast", column)
            kept = lookup(df, month_name, cls, "overcast_noprecip", column)
            # Bottom to top: kept, removed by precipitation, removed by overcast.
            # Medians of nested stages keep this order, so no height is negative.
            segments = (
                (0.0, kept, COLOR_BOTH, "white"),
                (kept, mid - kept, COLOR_OVERCAST, INK_PRIMARY),
                (mid, total - mid, COLOR_ANY_CLOUD, INK_PRIMARY),
            )
            for bottom, height, color, ink in segments:
                ax.bar(x[m, c], height, width, bottom=bottom, color=color,
                       edgecolor="white", linewidth=1.0)
                if height >= min_label_h:
                    ax.text(x[m, c], bottom + height / 2, format(height, label_format),
                            ha="center", va="center", fontsize=8.5, color=ink)
            share = f"\n{100 * kept / total:.0f}% kept" if total > 0 else ""
            ax.text(x[m, c], total + 0.01 * y_max, f"{format(total, label_format)}{share}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_PRIMARY, linespacing=1.15)
    style_axes(ax, x, centres)
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.set_ylabel(ylabel, color=INK_PRIMARY, fontsize=11)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK_PRIMARY)
    ax.legend(
        handles=[
            Patch(color=COLOR_ANY_CLOUD, label="Removed by overcast filter (tcc ≥ 0.95)"),
            Patch(color=COLOR_OVERCAST, label="Then removed by no-precipitation filter (tp < 0.05 mm/hr)"),
            Patch(color=COLOR_BOTH, label="Kept after both filters"),
        ],
        loc="upper right", frameon=False, fontsize=9.5,
    )
    fig.text(0.01, 0.005,
             f"Top number = median with cloud present (tcc > 0). Segments are differences between "
             f"medians. Hours with no cells of a class count as zero. {WEIGHT_NOTES[weighting]}",
             fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=350)
    plt.close(fig)
    return path


def plot_lwp(lwp_df: pd.DataFrame, path: Path, n_seasons: int, weighting: str) -> Path:
    """Figure 2: side-by-side bars, because mean LWP does not add across stages."""
    x, centres = bar_positions()
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    width = 0.27                                    # three bars fit inside one class slot
    offsets = (-width, 0.0, width)
    top = 0.0
    notes = []
    for m, month_name in enumerate(MONTHS.values()):
        for c, cls in enumerate(FIGURE_CLASSES):
            for offset, stage in zip(offsets, STAGES):
                value = lookup(lwp_df, month_name, cls, stage, "median_of_seasons")
                if not np.isfinite(value):
                    continue
                top = max(top, value)
                ax.bar(x[m, c] + offset, value, width, color=STAGE_COLORS[stage],
                       edgecolor="white", linewidth=0.8)
                ax.text(x[m, c] + offset, value + 1.0, f"{value:.0f}", ha="center", va="bottom",
                        fontsize=8, color=INK_PRIMARY)
            n_used = int(lookup(lwp_df, month_name, cls, "overcast_noprecip", "n_seasons"))
            if n_used < n_seasons:
                notes.append(f"{FIGURE_CLASSES[cls]} in {month_name}: {n_used} of {n_seasons} seasons")
    style_axes(ax, x, centres)
    ax.set_ylim(0, top * 1.14)
    ax.set_ylabel("Median of monthly mean LWP [g m⁻²]", color=INK_PRIMARY, fontsize=11)
    ax.set_title(
        f"LWP in supercooled liquid-containing cloud hours, by filter (median of {n_seasons} seasons)",
        loc="left", fontsize=12.5, color=INK_PRIMARY,
    )
    ax.legend(
        handles=[Patch(color=STAGE_COLORS[s], label=STAGE_LABELS[s]) for s in STAGES],
        loc="upper right", frameon=False, fontsize=9.5,
    )
    footer = f"Only hours passing each filter are averaged. {WEIGHT_NOTES[weighting]}"
    if notes:
        footer += " " + "; ".join(notes) + "."
    fig.text(0.01, 0.005, footer, fontsize=8.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=350)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------
def output_names(span: str, weighting: str) -> dict[str, str]:
    """File names; weighted_names tags each stem when area weighting is on."""
    tag = "liquidContaining_filterStages"
    month = f"ERA5_avgHoursPerCell_perMonth_{span}_{tag}"
    day = f"ERA5_avgHoursPerCell_perDay_{span}_{tag}"
    lwp = f"ERA5_LWP_perMonth_{span}_{tag}"
    names = {
        "hours_csv": f"{month}.csv",
        "lwp_csv": f"{lwp}.csv",
        "daily_csv": f"{day}_allDays.csv",
        "daily_summary_csv": f"{day}_summary.csv",
        "hours_fig": f"{month}.png",
        "daily_fig": f"{day}.png",
        "lwp_fig": f"{lwp}.png",
    }
    return {key: ov.weighted_names(name, weighting) for key, name in names.items()}


def find_span(out_dir: Path, weighting: str) -> str:
    """The season span tag of the hours table already in ``out_dir``."""
    prefix = ov.weighted_names("ERA5_avgHoursPerCell_perMonth_", weighting)
    suffix = "_liquidContaining_filterStages.csv"
    found = sorted(out_dir.glob(f"{prefix}*{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one {prefix}<span>{suffix} in {out_dir}, found {len(found)}; "
            "run without --plot-only first"
        )
    return found[0].name[len(prefix) : -len(suffix)]


def write_readme(out_dir: Path, names: dict[str, str], n_seasons: int, weighting: str) -> Path:
    """Describe the outputs, using the docstring as the definitions."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    text = (
        f"Generated {stamp} by make_era5_filter_stage_hours_lwp.py (repo {git_describe()})\n"
        f"Seasons: {n_seasons}. Cell weighting: {weighting}.\n\n"
        f"{names['hours_csv']}\n    Hours per cell by month, class, stage and season [h]\n"
        f"{names['lwp_csv']}\n    Mean LWP by month, class, stage and season [g m-2]\n"
        f"{names['daily_csv']}\n    Hours per cell for every day, class and stage [h]\n"
        f"{names['daily_summary_csv']}\n    Median and mean of those days by month [h]\n"
        f"{names['hours_fig']}\n    Stacked hours per month, median of seasons\n"
        f"{names['daily_fig']}\n    Stacked hours per day, median of days\n"
        f"{names['lwp_fig']}\n    LWP, side-by-side bars, median of seasons\n\n"
        f"{__doc__}"
    )
    path = out_dir / "readME.txt"
    path.write_text(text)
    return path


def draw_all(out_dir: Path, span: str, weighting: str) -> list[Path]:
    """All three figures, from the CSVs only, so --plot-only and a full run agree."""
    names = output_names(span, weighting)
    hours_df = pd.read_csv(out_dir / names["hours_csv"])
    lwp_df = pd.read_csv(out_dir / names["lwp_csv"])
    summary_df = pd.read_csv(out_dir / names["daily_summary_csv"])
    n_seasons = len([c for c in hours_df.columns if c not in NON_SEASON_COLUMNS])
    return [
        plot_stacked(
            hours_df, "median_of_seasons", out_dir / names["hours_fig"],
            f"Supercooled liquid-containing cloud hours per month, by filter (median of {n_seasons} seasons)",
            "Median hours per month, per grid cell", weighting,
            y_max=800.0, y_ticks=np.arange(0.0, 701.0, 100.0), label_format=".0f",
            min_label_h=MIN_SEGMENT_LABEL_H,
        ),
        plot_stacked(
            summary_df, "median_of_days", out_dir / names["daily_fig"],
            f"Supercooled liquid-containing cloud hours per day, by filter (median of all days, {n_seasons} seasons)",
            "Median hours per day, per grid cell", weighting,
            y_max=30.0, y_ticks=np.arange(0.0, 25.0, 6.0), label_format=".1f",
            min_label_h=MIN_SEGMENT_LABEL_H_DAY,
        ),
        plot_lwp(lwp_df, out_dir / names["lwp_fig"], n_seasons, weighting),
    ]


def run_one_weighting(opts: argparse.Namespace, run, weighting: str) -> None:
    """Read the raw fields, write the tables and draw the figures for one weighting."""
    hour_count_dir = run.out_dir(OUT_PRODUCT)
    out_dir = opts.out_dir or hour_count_dir / f"filter_stages{ov.weight_suffix(weighting)}"
    if opts.plot_only:
        for path in draw_all(out_dir, find_span(out_dir, weighting), weighting):
            print(f"  -> {path}")
        return

    print(f"\n=== cell weighting: {weighting} ===")
    args = build_args({**SETTINGS, **run.settings(), "cell_weighting": weighting})
    A = accumulate(args)
    span = ov.span_tag(A["seasons"])
    hours_df, lwp_df, daily_df, summary_df = build_tables(A)
    check_against_csvs(hours_df, daily_df, span, hour_count_dir, weighting)

    out_dir.mkdir(parents=True, exist_ok=True)
    names = output_names(span, weighting)
    hours_df.to_csv(out_dir / names["hours_csv"], index=False, float_format="%.4f")
    lwp_df.to_csv(out_dir / names["lwp_csv"], index=False, float_format="%.4f")
    daily_df.to_csv(out_dir / names["daily_csv"], index=False, float_format="%.4f")
    summary_df.to_csv(out_dir / names["daily_summary_csv"], index=False, float_format="%.4f")
    write_readme(out_dir, names, len(A["seasons"]), weighting)
    for path in draw_all(out_dir, span, weighting):
        print(f"  -> {path}")

    # The numbers behind the figures, for the log.
    for label, df, column in (
        ("Hours per month [h], median of seasons", hours_df, "median_of_seasons"),
        ("Hours per day [h], median of days", summary_df, "median_of_days"),
        ("Hours per day [h], median of days, without Oct-Dec 2020", summary_df, "median_excl_arm_missing"),
        ("Monthly mean LWP [g m-2], median of seasons", lwp_df, "median_of_seasons"),
    ):
        shown = df[df["class"].isin(FIGURE_CLASSES)]
        print(f"\n{label}")
        print(shown.pivot_table(index=["month", "class"], columns="stage",
                                values=column, sort=False)[list(STAGES)]
              .round(2).to_string())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory for one weighting (default <hour-count folder>/filter_stages)",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="redraw the figures from the CSVs already in the output folders",
    )
    p.add_argument(
        "--cell-weighting", choices=("both", "area", "uniform"), default="both",
        help="which version to make (default both, each in its own folder)",
    )
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    if opts.out_dir is not None and opts.cell_weighting == "both":
        p.error("--out-dir needs --cell-weighting area or uniform")
    run = ov.run_from_args(opts)
    if not opts.plot_only:
        print(run.report())
    weightings = ("area", "uniform") if opts.cell_weighting == "both" else (opts.cell_weighting,)
    for weighting in weightings:
        run_one_weighting(opts, run, weighting)


if __name__ == "__main__":
    main()
