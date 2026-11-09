#!/usr/bin/env python3
"""ERA5 clouds over open water and over sea ice, October and November.

THE QUESTION
============
Satellite lidar studies find more cloud over open water than over sea ice in
fall (Palm et al. 2010, doi:10.1029/2010JD013900; Morrison et al. 2018,
doi:10.1002/2017JD027248), while the filtered ERA5 hours in this folder show
more liquid-containing hours over pack ice by 31 October. This script shows
where the two part ways, by computing the same comparison for a ladder of
metrics, from "is there any cloud" to the filtered hours.

TWO COMPARISONS, SIDE BY SIDE
=============================
1. Class mean. Each hour, the average over the cells of that surface; then the
   average over the hours the surface existed. This is the per_hour method of
   make_era5_class_per_cell_csvs.py, so these are the numbers in those tables.

2. Same cell, same half month. Open water and ice mix geography and calendar:
   in this domain open cells lie south of ice cells, and a cell tends to be open
   early and icy late. So, for each grid cell and each half month (1-15 and 16 to
   the end of the month), water and ice are averaged separately over all years,
   and the two are differenced. Only cell-periods with at least MIN_PAIR_HOURS of
   each surface count, and those are then averaged. This follows the idea of the
   intermittent mask in Morrison et al. (2018, section 3.1): compare the two
   surfaces only where the same place has seen both. It does NOT control for
   weather regime.

The uncertainty on (2) is a leave-one-season-out jackknife, because hours within
a season are not independent. Nothing here is random, so there is no seed.

TWO SETS OF SURFACE THRESHOLDS
==============================
    project    open water: siconc < 0.05;  ice: siconc > 0.95   (the table classes)
    morrison   open water: siconc < 0.15;  ice: siconc > 0.80   (Morrison section 3.2)

Both use pure sea cells only (land fraction <= lsm_tol), so they exclude the
coastal cells, whose class ignores sea ice, and the land cells.

THE METRICS
===========
Phase and filters are the project's own (make_era5_hour_count_csvs.py):
liquid containing means IWP/CWP < 0.90 with species at or below 0.01 g m-2
counted as absent; "overcast" means tcc >= 0.95; "precipitating" means
tp >= 0.05 mm hr-1. See METRICS below for each one.

WHAT THIS CANNOT SAY
====================
ERA5 total cloud cover is not what a lidar reports, so these numbers are not
directly comparable with CALIPSO values without a lidar simulator. Liquid cloud
fraction in Morrison et al. is per height level, while liquid containing here is
per column. "Supercooled" is not tested: nothing here looks at temperature.

Usage
-----
    python make_era5_ice_vs_water_clouds.py               # about 3 min
    python make_era5_ice_vs_water_clouds.py --plot-only   # redraw from the CSV
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ov_config as ov  # first: thread limits and sys.path on Casper

from make_era5_hour_count_csvs import (
    OUT_PRODUCT,
    SETTINGS,
    build_args,
    git_describe,
    season_label,
)

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from cloud_classification import fraction_phase_masks  # noqa: E402
from seb_analysis_common import load_seb_data, resolve_region_dir  # noqa: E402
from surface_classification import (  # noqa: E402
    CLASS_COLORS,
    align_lsm_to_grid,
    area_weights_2d,
    iter_time_blocks,
    load_land_sea_mask,
)

# ----------------------------------------------------------------------------
# Definitions
# ----------------------------------------------------------------------------
MIN_PAIR_HOURS = 72     # h; each surface needs this many in a cell-period

# (name, open water below this siconc, ice above this siconc)
THRESHOLDS = (("project", 0.05, 0.95), ("morrison", 0.15, 0.80))
THRESHOLD_WORDS = {
    "project": "open water: sea ice < 5%,  pack ice: sea ice > 95%",
    "morrison": "open water: sea ice < 15%,  ice: sea ice > 80% (Morrison et al. 2018)",
}
# Half-month bins: index = 2 * (month is November) + (day is after the 15th).
PERIODS = ("Oct 1-15", "Oct 16-31", "Nov 1-15", "Nov 16-30")
MONTHS = {"October": (0, 1), "November": (2, 3), "October+November": (0, 1, 2, 3)}

# Key -> what it measures. All are fractions of hours, 0-1, except the three
# means: tcc and lcc are fractions of the sky, and the water paths are g m-2.
METRICS = {
    "tcc_mean": "mean total cloud cover",
    "lcc_mean": "mean low cloud cover",
    "tcc_ge_0p5": "tcc >= 0.5",
    "overcast": "tcc >= 0.95, any condensate",
    "precipitating": "tp >= 0.05 mm hr-1",
    "lc_any": "liquid containing, no filters",
    "lc_overcast": "liquid containing, overcast filter only",
    "lc_noprecip": "liquid containing, no-precipitation filter only",
    "lc_kept": "liquid containing, both filters (the table quantity)",
    "lwp_gt_1": "LWP > 1 g m-2, no filters, no phase test",
    "lwp_gt_10": "LWP > 10 g m-2, no filters, no phase test",
    "lwp_gt_30": "LWP > 30 g m-2, no filters, no phase test",
    "lwp_mean_g_m2": "mean LWP including zeros (g m-2)",
    "iwp_mean_g_m2": "mean IWP including zeros (g m-2)",
}
METRIC_KEYS = tuple(METRICS)

# Rows of the figure: (group heading, [(metric key, row label), ...]). The mean
# water paths have a different unit, so they are stated in the panel title and
# not drawn on the fraction axis.
FIGURE_ROWS = (
    ("Cloud amount", (("tcc_mean", "Mean total cloud cover"),
                      ("lcc_mean", "Mean low cloud cover"),
                      ("overcast", "Overcast (tcc $\\geq$ 0.95)"))),
    ("Precipitation", (("precipitating", "tp $\\geq$ 0.05 mm hr$^{-1}$"),)),
    ("Liquid-containing column", (("lc_any", "No filters"),
                                  ("lc_overcast", "Overcast filter only"),
                                  ("lc_noprecip", "No-precipitation filter only"),
                                  ("lc_kept", "Both filters (the tables)"))),
    ("Liquid amount, no filters", (("lwp_gt_1", "LWP > 1 g m$^{-2}$"),
                                   ("lwp_gt_10", "LWP > 10 g m$^{-2}$"),
                                   ("lwp_gt_30", "LWP > 30 g m$^{-2}$"))),
)

# Figure styling. The surfaces keep their colours from the other class figures.
# Checked for colour-blind separation with OKLab distance x 100 after Machado et
# al. (2009) simulation at full severity: normal 26, protan 24, deutan 27,
# tritan 23 (a target of >= 8 is common).
COLOR_WATER = CLASS_COLORS["open_ocean"]
COLOR_ICE = CLASS_COLORS["sea_ice"]
INK_PRIMARY = "#222222"    # titles, row labels and difference markers
INK_MUTED = "#666666"      # tick labels and printed values
GRID_COLOR = "#e4e4e4"     # recessive gridlines and connectors
BAND_COLOR = "#f1f1f1"     # the row that is the table quantity


# ----------------------------------------------------------------------------
# The streaming pass
# ----------------------------------------------------------------------------
def period_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Half-month bin of each time step (see PERIODS)."""
    return 2 * (times.month == 11).astype(int) + (times.day > 15).astype(int)


def accumulate(args: argparse.Namespace, seasons: list[int]) -> dict:
    """Read every October and November hour and accumulate each metric.

    Returns, with n_surf = 2 * len(THRESHOLDS) and water before ice:

    hour_sum (period, surf, metric)  sum over hours of the hourly class mean
    hour_n   (period, surf)          hours in which the surface existed
    acc      (season, period, surf, lat, lon, metric)  sum over hours per cell
    cnt      (season, period, surf, lat, lon)          hours per cell
    cell_w   (lat, lon)              weight of each cell in every average
    """
    ds = load_seb_data(args.region, None, None, resolve_region_dir(args).parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(
            args.region, lwph.resolve_data_root(args.storage, args.data_root), args.mask_grid
        ),
        ds,
    )
    is_sea = lsm <= args.lsm_tol                                # (lat, lon)
    # One cell, one vote unless area weighting was asked for.
    cell_w = (
        area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])
        if args.cell_weighting == "area"
        else np.ones(lsm.shape)
    )

    times = pd.DatetimeIndex(ds["valid_time"].values)
    # A season's October and November fall in its own start year.
    keep = np.isin(times.month, (10, 11)) & np.isin(times.year, seasons)

    n_y, n_p, n_s, n_m = len(seasons), len(PERIODS), 2 * len(THRESHOLDS), len(METRIC_KEYS)
    hour_sum = np.zeros((n_p, n_s, n_m))
    hour_n = np.zeros((n_p, n_s))
    acc = np.zeros((n_y, n_p, n_s, *lsm.shape, n_m))
    cnt = np.zeros((n_y, n_p, n_s, *lsm.shape))
    print(f"  Reading {int(keep.sum()):,} hours; cell tally {acc.nbytes / 1e6:,.0f} MB", flush=True)

    read = ["tcc", "lcc", args.liquid_var, "tciw", "tp", "siconc"]
    for i0, block in iter_time_blocks(ds, read, args.block_hours, keep_mask=keep):
        span = slice(i0, i0 + block.sizes["valid_time"])
        sel = keep[span]
        if not sel.any():
            continue
        block_times = times[span][sel]
        year_idx = np.searchsorted(seasons, block_times.year)   # (t,)
        period = period_index(block_times)                      # (t,)

        # All (t, lat, lon).
        tcc = block["tcc"].values[sel]                          # fraction 0-1
        lcc = block["lcc"].values[sel]                          # fraction 0-1
        lwp_g_m2 = block[args.liquid_var].values[sel] * 1000.0  # kg m-2 -> g m-2
        iwp_g_m2 = block["tciw"].values[sel] * 1000.0           # kg m-2 -> g m-2
        tp_mm_hr = block["tp"].values[sel] * 1000.0             # m hr-1 -> mm hr-1
        sic = block["siconc"].values[sel]                       # fraction 0-1, NaN on land

        # The same filter and phase test as make_era5_hour_count_csvs.py.
        valid = np.isfinite(tcc) & np.isfinite(lwp_g_m2) & np.isfinite(iwp_g_m2)
        phase = fraction_phase_masks(
            lwp_g_m2, iwp_g_m2, args.liquid_fraction_min, args.ice_fraction_min,
            args.min_lwp, args.min_iwp,
        )
        liquid_containing = phase["liquid"] | phase["mixed"]
        overcast = tcc >= args.min_cloud_fraction
        raining = np.isfinite(tp_mm_hr) & (tp_mm_hr >= args.precip_rate_max)

        fields = {
            "tcc_mean": tcc,
            "lcc_mean": lcc,
            "tcc_ge_0p5": tcc >= 0.5,
            "overcast": overcast,
            "precipitating": raining,
            "lc_any": liquid_containing,
            "lc_overcast": liquid_containing & overcast,
            "lc_noprecip": liquid_containing & ~raining,
            "lc_kept": liquid_containing & overcast & ~raining,
            "lwp_gt_1": lwp_g_m2 > 1.0,
            "lwp_gt_10": lwp_g_m2 > 10.0,
            "lwp_gt_30": lwp_g_m2 > 30.0,
            "lwp_mean_g_m2": lwp_g_m2,
            "iwp_mean_g_m2": iwp_g_m2,
        }

        surfaces = []
        with np.errstate(invalid="ignore"):
            for _name, open_max, ice_min in THRESHOLDS:
                surfaces += [is_sea & valid & (sic < open_max), is_sea & valid & (sic > ice_min)]

        for si, mask in enumerate(surfaces):                    # mask (t, lat, lon)
            weight_t = (mask * cell_w).sum(axis=(1, 2))         # (t,)
            exists = weight_t > 0
            np.add.at(hour_n[:, si], period[exists], 1.0)
            for ym in np.unique(np.stack([year_idx, period]), axis=1).T:
                in_ym = (year_idx == ym[0]) & (period == ym[1])
                cnt[ym[0], ym[1], si] += mask[in_ym].sum(axis=0)
            for mi, key in enumerate(METRIC_KEYS):
                value = np.where(mask, np.asarray(fields[key], dtype=float), 0.0)
                # NaN in a field other than tcc, tclw, tciw would poison a sum.
                value = np.nan_to_num(value)
                class_mean = (value * cell_w).sum(axis=(1, 2))[exists] / weight_t[exists]
                np.add.at(hour_sum[:, si, mi], period[exists], class_mean)
                for ym in np.unique(np.stack([year_idx, period]), axis=1).T:
                    in_ym = (year_idx == ym[0]) & (period == ym[1])
                    acc[ym[0], ym[1], si, :, :, mi] += value[in_ym].sum(axis=0)

    return {"hour_sum": hour_sum, "hour_n": hour_n, "acc": acc, "cnt": cnt, "cell_w": cell_w}


# ----------------------------------------------------------------------------
# The two comparisons
# ----------------------------------------------------------------------------
def same_cell_means(R: dict, si_water: int, periods: tuple, season_mask: np.ndarray) -> tuple:
    """Water and ice means (metric,) over eligible cell-periods, and how many there are."""
    idx = list(periods)
    # Sum over the kept seasons: (period, lat, lon, metric) and (period, lat, lon).
    acc_w = R["acc"][season_mask][:, idx, si_water].sum(axis=0)
    acc_i = R["acc"][season_mask][:, idx, si_water + 1].sum(axis=0)
    cnt_w = R["cnt"][season_mask][:, idx, si_water].sum(axis=0)
    cnt_i = R["cnt"][season_mask][:, idx, si_water + 1].sum(axis=0)
    ok = (cnt_w >= MIN_PAIR_HOURS) & (cnt_i >= MIN_PAIR_HOURS)   # (period, lat, lon)
    w = np.broadcast_to(R["cell_w"], ok.shape)[ok]               # (n_ok,)
    water = acc_w[ok] / cnt_w[ok][:, None]                       # (n_ok, metric)
    ice = acc_i[ok] / cnt_i[ok][:, None]                         # (n_ok, metric)
    return np.average(water, axis=0, weights=w), np.average(ice, axis=0, weights=w), int(ok.sum())


def build_table(R: dict) -> pd.DataFrame:
    """One row per (thresholds, month, metric), holding both comparisons."""
    n_seasons = R["acc"].shape[0]
    rows = []
    for ti, (name, _open_max, _ice_min) in enumerate(THRESHOLDS):
        si_water = 2 * ti
        for month, periods in MONTHS.items():
            idx = list(periods)
            # Class means, each hour counting equally.
            class_water = R["hour_sum"][idx, si_water].sum(axis=0) / R["hour_n"][idx, si_water].sum()
            class_ice = (
                R["hour_sum"][idx, si_water + 1].sum(axis=0) / R["hour_n"][idx, si_water + 1].sum()
            )
            # Same cell, same half month, from all seasons.
            everything = np.ones(n_seasons, dtype=bool)
            water, ice, n_pairs = same_cell_means(R, si_water, periods, everything)

            # Leave-one-season-out jackknife of the difference.
            diffs = []
            for y in range(n_seasons):
                without_y = everything.copy()
                without_y[y] = False
                w_y, i_y, _ = same_cell_means(R, si_water, periods, without_y)
                diffs.append(w_y - i_y)
            diffs = np.array(diffs)                              # (season, metric)
            se = np.sqrt((n_seasons - 1) / n_seasons * ((diffs - diffs.mean(axis=0)) ** 2).sum(axis=0))

            for mi, key in enumerate(METRIC_KEYS):
                rows.append({
                    "thresholds": name, "period": month, "metric": key,
                    "description": METRICS[key],
                    "class_mean_water": class_water[mi], "class_mean_ice": class_ice[mi],
                    "same_cell_water": water[mi], "same_cell_ice": ice[mi],
                    "same_cell_water_minus_ice": water[mi] - ice[mi],
                    "same_cell_jackknife_se": se[mi],
                    "n_cell_periods": n_pairs,
                })
    return pd.DataFrame(rows)


def check_against_cumulative(R: dict, n_seasons: int, cum_dir: Path, span: str,
                             weighting: str) -> None:
    """The filtered class means must reproduce the cumulative table at month ends.

    Summed over hours and divided by the number of seasons, lc_kept is the average
    cumulative hours per cell, which that table states on 31 Oct and 30 Nov.
    """
    # The cumulative script tags area-weighted file names ("..._areaWeighted_perDay_..."),
    # so the name must carry the same tag or the check silently finds nothing.
    path = cum_dir / ov.weighted_names(
        f"ERA5_cumulativeHoursPerCell_perDay_{span}_liquidContaining.csv", weighting)
    if not path.exists():
        print(f"  !! no table to check against: {path.name}")
        return
    cum = pd.read_csv(path, index_col="date")
    mi = METRIC_KEYS.index("lc_kept")
    worst_h = 0.0
    # Surfaces 0 and 1 are the project thresholds, the table's two classes.
    for si, column in enumerate(("OpenOcean", "PackIce")):
        for date, last_period in (("31 Oct", 2), ("30 Nov", 4)):
            mine_h = R["hour_sum"][:last_period, si, mi].sum() / n_seasons
            worst_h = max(worst_h, abs(mine_h - cum.loc[date, column]))
    if worst_h > 1.0e-3:
        raise AssertionError(f"class means differ from {path.name} by {worst_h:.5f} h")
    print(f"  matches the cumulative table on 31 Oct and 30 Nov (worst {worst_h:.5f} h)")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def plot_figure(table: pd.DataFrame, thresholds: str, out_path: Path, region: str, span: str) -> Path:
    """Dumbbells of the class means beside the same-cell differences, by month."""
    # Agg because the script also runs on Casper compute nodes with no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = table[table["thresholds"] == thresholds].set_index(["period", "metric"])

    # Lay the rows out top down, leaving one row for each group heading.
    y_rows, y_headings, y = [], [], 0.0
    for heading, rows in FIGURE_ROWS:
        y_headings.append((y, heading))
        y -= 1.0
        for key, label in rows:
            y_rows.append((y, key, label))
            y -= 1.0
        y -= 0.45      # a little air between the groups
    y_min, y_max = y + 0.45 - 0.6, 0.6

    # One difference axis for both months, symmetric so zero sits in the middle.
    keys = [key for _, key, _ in y_rows]
    use = df.loc[(list(("October", "November")), keys), :]
    x_diff = float(
        (use["same_cell_water_minus_ice"].abs() + use["same_cell_jackknife_se"]).max()
    ) * 1.5

    fig, axes = plt.subplots(
        1, 4, figsize=(14.0, 8.0), sharey=True,
        gridspec_kw={"width_ratios": [1.5, 1.15, 1.5, 1.15], "wspace": 0.08},
    )

    for col, month in enumerate(("October", "November")):
        ax_dumb, ax_diff = axes[2 * col], axes[2 * col + 1]

        for ax in (ax_dumb, ax_diff):
            ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(GRID_COLOR)
            ax.tick_params(axis="x", colors=INK_MUTED, length=0, labelsize=9)
            ax.tick_params(axis="y", length=0)
            # The row that is the table quantity, across all four panels.
            for row_y, key, _label in y_rows:
                if key == "lc_kept":
                    ax.axhspan(row_y - 0.5, row_y + 0.5, color=BAND_COLOR, zorder=0, linewidth=0)

        ax_diff.axvline(0.0, color=INK_MUTED, linewidth=1.0, zorder=1)

        for row_y, key, _label in y_rows:
            r = df.loc[(month, key)]
            water, ice = r["class_mean_water"], r["class_mean_ice"]
            # --- class means: the gap between the two surfaces ----------------
            ax_dumb.plot([water, ice], [row_y, row_y], color="#bbbbbb", linewidth=2.0, zorder=2)
            ax_dumb.scatter(water, row_y, s=62, color=COLOR_WATER, zorder=4,
                            edgecolor="white", linewidth=0.8)
            ax_dumb.scatter(ice, row_y, s=62, color=COLOR_ICE, zorder=3,
                            edgecolor="white", linewidth=0.8)
            # Printed outward from the pair, so the two never overlap.
            for value, is_upper in ((water, water >= ice), (ice, ice > water)):
                ax_dumb.annotate(
                    f"{value:.2f}", (value, row_y), xytext=(8 if is_upper else -8, 0),
                    textcoords="offset points", va="center",
                    ha="left" if is_upper else "right", fontsize=8.5, color=INK_MUTED,
                )

            # --- same cell: the difference and its uncertainty ----------------
            diff, se = r["same_cell_water_minus_ice"], r["same_cell_jackknife_se"]
            clear_of_zero = abs(diff) > 2.0 * se
            ax_diff.plot([diff - se, diff + se], [row_y, row_y], color=INK_PRIMARY, linewidth=1.4,
                         zorder=2, solid_capstyle="butt")
            # Small enough that a +/-1 SE line of about 0.03 still shows past it.
            ax_diff.scatter(diff, row_y, s=22, zorder=3, linewidth=1.1, edgecolor=INK_PRIMARY,
                            color=INK_PRIMARY if clear_of_zero else "white")
            ax_diff.annotate(
                f"{diff:+.2f}", (diff + se if diff >= 0 else diff - se, row_y),
                xytext=(5 if diff >= 0 else -5, 0), textcoords="offset points", va="center",
                ha="left" if diff >= 0 else "right", fontsize=8.5, color=INK_MUTED,
            )

        lwp_w = df.loc[(month, "lwp_mean_g_m2"), "class_mean_water"]
        lwp_i = df.loc[(month, "lwp_mean_g_m2"), "class_mean_ice"]
        ax_dumb.set_title(
            f"{month}: class mean\nMean LWP {lwp_w:.0f} water, {lwp_i:.0f} ice (g m$^{{-2}}$)",
            loc="left", fontsize=10.5, color=INK_PRIMARY,
        )
        n_pairs = int(df.loc[(month, "tcc_mean"), "n_cell_periods"])
        ax_diff.set_title(
            f"Same cell, same half month\n({n_pairs:,} cell-periods)",
            loc="left", fontsize=10.5, color=INK_PRIMARY,
        )
        ax_dumb.set_xlim(-0.04, 1.14)
        ax_dumb.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_dumb.set_xlabel("Fraction of hours or of sky", fontsize=10, color=INK_PRIMARY)
        ax_diff.set_xlim(-x_diff, x_diff)
        ax_diff.set_xlabel("Water $-$ ice", fontsize=10, color=INK_PRIMARY)

    # Row labels on the left panel only, with group headings in bold.
    ticks = [(row_y, label, False, key == "lc_kept") for row_y, key, label in y_rows]
    ticks += [(row_y, heading, True, False) for row_y, heading in y_headings]
    ticks.sort(key=lambda t: -t[0])
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_yticks([t[0] for t in ticks])
    axes[0].set_yticklabels([t[1] for t in ticks], fontsize=10, color=INK_PRIMARY)
    for label, (_, _, is_heading, is_table) in zip(axes[0].get_yticklabels(), ticks):
        if is_heading or is_table:
            label.set_fontweight("bold")

    first, last = int(span.split("_")[0]), int(span.split("_")[1]) - 1
    fig.suptitle(
        f"ERA5 clouds over open water and pack ice — {region}, "
        f"{season_label(first)} to {season_label(last)}\n{THRESHOLD_WORDS[thresholds]}",
        fontsize=13, color=INK_PRIMARY, y=0.985,
    )
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=COLOR_WATER, label="Open water"),
        Line2D([], [], marker="o", linestyle="", markersize=8, color=COLOR_ICE, label="Pack ice"),
        Line2D([], [], marker="o", color=INK_PRIMARY, markersize=7, linewidth=1.4,
               label="Difference, more than 2 SE from zero"),
        Line2D([], [], marker="o", color=INK_PRIMARY, markerfacecolor="white", markersize=7,
               linewidth=1.4, label="Difference, within 2 SE (line = $\\pm$1 SE)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
               labelcolor=INK_PRIMARY)
    fig.subplots_adjust(top=0.86, bottom=0.135, left=0.165, right=0.99)
    fig.savefig(out_path, dpi=350, facecolor="white")    # 350 dpi, as the other OV figures
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------
def write_readme(out_dir: Path, span: str, weighting: str) -> None:
    words = "cos(latitude)" if weighting == "area" else "not weighted (each grid cell counts equally)"
    text = f"""ERA5 CLOUDS OVER OPEN WATER AND OVER SEA ICE, OCTOBER AND NOVEMBER
==================================================================

Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_ice_vs_water_clouds.py
(repo {git_describe()}). Averages are {words}. Data, filters and phase are as in
../readME.txt. The method is in the script's docstring.

  ERA5_iceVsWater_{span}.csv
      one row per thresholds x period x metric:
        class_mean_water, class_mean_ice      hourly class means, each hour equal;
                                              lc_kept equals the table quantity
        same_cell_water, same_cell_ice        same grid cell, same half month
        same_cell_water_minus_ice             their difference
        same_cell_jackknife_se                leave-one-season-out standard error
        n_cell_periods                        how many cell-periods were eligible
                                              (at least {MIN_PAIR_HOURS} h of each surface)
      thresholds: project (sea ice < 5%, > 95%); morrison (< 15%, > 80%).

  ERA5_iceVsWater_{span}_<thresholds>.png
      the figure, for October and November.

Fractions are of hours, except mean total and low cloud cover, which are of the
sky. ERA5 cloud cover is not what a lidar reports, and liquid containing here is
per column, not per height level as in Morrison et al. (2018).
"""
    (out_dir / "readME.txt").write_text(text)


def find_span(out_dir: Path) -> str:
    """The season span tag of the table already in ``out_dir``."""
    prefix, suffix = "ERA5_iceVsWater_", ".csv"
    found = sorted(out_dir.glob(f"{prefix}*{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one {prefix}<span>{suffix} in {out_dir}, found {len(found)}; "
            "run without --plot-only first"
        )
    return found[0].name[len(prefix) : -len(suffix)]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default <hour-count folder>/ice_vs_water)")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the figures from the CSV in the output folder, no reading of ERA5")
    ov.add_weighting_arg(p)
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    hour_count_dir = run.out_dir(OUT_PRODUCT)
    suffix = ov.weight_suffix(opts.cell_weighting)
    out_dir = opts.out_dir or hour_count_dir / f"ice_vs_water{suffix}"
    args = build_args({**SETTINGS, **run.settings(), "cell_weighting": opts.cell_weighting})

    if not opts.plot_only:
        print("=" * 72)
        print("ERA5 clouds over open water and over sea ice, October and November")
        print(run.report())
        print("=" * 72)
        seasons = sorted(int(y) for y in run.years)
        span = ov.span_tag(seasons)
        R = accumulate(args, seasons)
        check_against_cumulative(
            R, len(seasons), hour_count_dir / f"cumulative_by_class{suffix}", span,
            opts.cell_weighting,
        )
        table = build_table(R)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / f"ERA5_iceVsWater_{span}.csv", index=False, float_format="%.4f")
        write_readme(out_dir, span, opts.cell_weighting)

        pd.set_option("display.width", 200)
        show = table[(table.thresholds == "project") & (table.period == "October")]
        print("\n  October, project thresholds:")
        print(show.drop(columns=["thresholds", "period", "description"]).round(3).to_string(index=False))

    span = find_span(out_dir)
    table = pd.read_csv(out_dir / f"ERA5_iceVsWater_{span}.csv")
    for name, _open_max, _ice_min in THRESHOLDS:
        out_path = out_dir / f"ERA5_iceVsWater_{span}_{name}.png"
        print(f"  -> {plot_figure(table, name, out_path, run.domain.name, span)}")


if __name__ == "__main__":
    main()
