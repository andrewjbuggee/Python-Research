#!/usr/bin/env python3
"""Monthly box plots of ERA5 cloud temperature for UTQ and three surface classes.

WHAT IT DRAWS
=============
One figure with October to March on the x axis and four boxes per month: UTQ,
Land, Open ocean and Pack ice. Each box is the distribution of the hourly values
in one ERA5_CloudTemp_*_perHour_*_<CLASS>.csv file, as written by
make_era5_hourly_values_csvs.py, for one sky state (liquid containing by
default) and one calendar month, pooled over every season in the file.

WHAT EACH VALUE IS
==================
From the readME beside those files: the condensate-weighted mean temperature of
the column, from ERA5 pressure-level data (23 levels, 1000-200 hPa),

    T = sum_l (q_l dp_l T_l) / sum_l (q_l dp_l)        [deg C]

with q = clwc for the liquid states and q = ciwc for ice only. It is the
temperature of the cloud's liquid (or ice), not cloud base or cloud top.

UTQ is the single grid cell containing the ARM site. For a surface class, each
hour's value is the mean over the cells of the class in that sky state in that
hour (cos-latitude weighted in the areaWeighted files). So, as the readME says:

  * every HOUR counts equally in a box, whether 1 cell or 1,000 cells were in
    the state; and
  * a class box is a distribution of spatial means, which should be narrower
    than the distribution over single grid cells (my inference, not measured).

THE NUMBERS UNDER THE BOXES
===========================
Hours per month in the sky state for a TYPICAL GRID CELL of the class, mean of
the seasons. Each hour's count of cells in the state is divided by the number
of cells in the class at that hour, and the quotients are summed over the month:

    H = sum_t n(t) / N(t)        [h]

This is the per_hour average of make_era5_class_per_cell_csvs.py. The hourly
CSVs do not carry N(t), so H is read from that script's monthly tables for the
same run (era5_hour_count_csv/class_normalized[_areaWeighted] beside --in-dir),
in the same weighting as the hourly files. UTQ is one cell, so its H is its own
hours, from ERA5_totalHours_perMonth_*_UTQ.csv.

H counts every hour in the state, including those with no cloud temperature
(a column with < 0.001 g m-2 of the weighting condensate on the pressure levels).
For the Rui-region files of 9 Nov 2026, the share of liquid-containing cell-hours
WITH a cloud temperature is: UTQ 93%, Land 99%, Open ocean 99.9%, Pack ice 84%
(cell_counts/). So the pack ice boxes rest on fewer cell-hours than H suggests.

H is not the number of values in the box. A class box holds one value for every
hour with at least one cell in the state, a number that is much larger than H
for a big class; the run prints both.

SMALL CLASSES ARE DROPPED
=========================
A class's box is not drawn in a month when the class covers less than 0.5% of
the domain on average (MIN_CLASS_SHARE_PCT), the same threshold at which
plot_surface_class_timeseries blanks a class's line. In the Rui region this drops
open ocean in January to March, when it averages at most 0.11% of the domain on
any day and its few hours come from one to five seasons.

The share is computed from normalization/ERA5_allValidCellHours_perMonth_*.csv
in the same run's hour-count folder: the class's cell-hours over all seasons,
divided by all cell-hours of the five classes in that month. It is a share of
CELLS, while the time series uses cos(latitude) AREA. The two differ by less
than a factor of 1.41 in a 65-80 N domain, which does not change which months
drop here (largest dropped share 0.03%; kept months 2.3% or more).

NO SUPERCOOLED FILTER
=====================
The Rui-region files of 9 Nov 2026 were made without the supercooled-only filter,
so liquid-containing hours can have T >= 0 deg C. The figure marks 0 deg C, and
the run prints the share of hours at or above it.

PLOTTING CONVENTIONS
====================
Layout as in plot_dlr_by_phase._draw_sky_state_boxes: black medians, hour counts
under the boxes, months with fewer than 30 hours hatched, 350 dpi.

Colours are Genie's ERA5 colours from the legend of her figures (UTQ red, Land
brown, Open ocean teal, Pack ice pink), in GENIE_REGION_COLORS. They were sampled
from a screenshot of that legend (12 Nov 2026), so they may differ by a few RGB
units from the values in her plotting code. As in her figures, the boxes are
filled at full opacity. Edges, whiskers and caps use a darker shade of the same
colour, so the pale pink box keeps a visible outline.

Whiskers: the default is 1.5 x IQR (Tukey), with no outliers drawn, so extreme
values neither appear nor stretch the whiskers. --whis minmax gives the
directory's other convention (whiskers to the minimum and maximum).

USAGE
=====
    python make_era5_cloud_temp_boxplots.py
    python make_era5_cloud_temp_boxplots.py --state ice_only
    python make_era5_cloud_temp_boxplots.py --in-dir <folder> --region-label "Barrow"

Nothing here is stochastic, so there is no random seed.
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

# First: puts ERA5/surface_energy_budget on sys.path for the colour constants.
import ov_config as ov

import matplotlib

matplotlib.use("Agg")  # also runs on Casper compute nodes, which have no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ----------------------------------------------------------------------------
# Inputs and what is drawn
# ----------------------------------------------------------------------------
DEFAULT_IN_DIR = (ov.HERE / "era5_hourly_values_csv_Rui_region_9_nov_2026"
                  / "era5_hourly_values_csv_areaWeighted")
DEFAULT_REGION_LABEL = "Rui region (65-80 N, 180-140 W)"

# Genie's ERA5 colours, sampled from the legend of her figures (see docstring).
GENIE_REGION_COLORS: dict[str, str] = {
    "UTQ": "#d02324",        # red
    "Land": "#8b6255",       # brown
    "OpenOcean": "#007575",  # teal
    "PackIce": "#ffadb9",    # pink
}
# File tag -> (legend label, colour), in Figure 7's panel order.
REGIONS: dict[str, tuple[str, str]] = {
    "UTQ": ("UTQ", GENIE_REGION_COLORS["UTQ"]),
    "Land": ("Land", GENIE_REGION_COLORS["Land"]),
    "OpenOcean": ("Open Ocean", GENIE_REGION_COLORS["OpenOcean"]),
    "PackIce": ("Pack Ice", GENIE_REGION_COLORS["PackIce"]),
}
EDGE_DARKEN = 0.65   # edge colour = fill RGB x this factor, a darker shade of the fill
STATES = {"liquid_containing": "liquid-containing", "liquid_only": "liquid-only",
          "ice_only": "ice-only"}
MONTHS = (10, 11, 12, 1, 2, 3)

# Full-opacity fills, as in Genie's figures; MIN_HOURS as plot_dlr_by_phase.
BOX_ALPHA = 1.0
MIN_HOURS = 30          # h in the whole record; fewer and the box is hatched
TUKEY_WHIS = 1.5        # whisker reach, in units of the interquartile range
MIN_CLASS_SHARE_PCT = 0.5   # % of domain cells; a smaller class-month is not drawn
CLASS_FILES = ("Land", "Coastal", "OpenOcean", "MarginalSeaIce", "PackIce")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def find_file(in_dir: Path, region: str) -> Path:
    """The CloudTemp file of one region, area-weighted or not."""
    found = sorted(in_dir.glob(f"ERA5_CloudTemp*_perHour_*_{region}.csv"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one CloudTemp file for {region} in {in_dir}, "
                                f"found {len(found)}")
    return found[0]


def load(in_dir: Path, state: str) -> tuple[dict, int]:
    """{region: {month: values [deg C]}} for one sky state, and the season count."""
    out, n_seasons = {}, 0
    for region in REGIONS:
        df = pd.read_csv(find_file(in_dir, region), usecols=["date", state])
        when = pd.to_datetime(df["date"])
        # Seasons start in October, so Jan-Mar belong to the previous year's season.
        season = np.where(when.dt.month >= 10, when.dt.year, when.dt.year - 1)
        n_seasons = max(n_seasons, len(np.unique(season)))
        out[region] = {m: df.loc[when.dt.month == m, state].dropna().to_numpy()
                       for m in MONTHS}
    return out, n_seasons


# File tag of each sky state in the hour-count tables.
STATE_TAGS = {"liquid_containing": "liquidContaining", "liquid_only": "liquidOnly",
              "ice_only": "iceOnly"}


def per_cell_hours(in_dir: Path, state: str) -> dict:
    """{region: {month: H}}: hours per month in ``state`` for a typical grid cell.

    H = sum_t n(t)/N(t), mean over seasons, from the monthly per-cell tables the
    same run wrote (see the docstring). Their weighting follows the hourly files.
    """
    count_dir = in_dir.parent / "era5_hour_count_csv"
    area = "areaWeighted" in find_file(in_dir, "UTQ").name
    class_dir = count_dir / ("class_normalized_areaWeighted" if area else "class_normalized")
    stem = "ERA5_avgHoursPerCell_areaWeighted_perMonth_" if area else "ERA5_avgHoursPerCell_perMonth_"
    tag = STATE_TAGS[state]
    out = {}
    for region in REGIONS:
        if region == "UTQ":
            found = sorted(count_dir.glob(f"ERA5_totalHours_perMonth_*_{tag}_UTQ.csv"))
        else:
            found = sorted(class_dir.glob(f"{stem}*_{tag}_{region}.csv"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one per-cell table for {region}, {state} "
                                    f"(in {class_dir if region != 'UTQ' else count_dir}), "
                                    f"found {len(found)}")
        table = pd.read_csv(found[0]).set_index("month")   # rows Oct..Mar, columns seasons
        out[region] = {m: float(table.loc[calendar.month_abbr[m]].mean()) for m in MONTHS}
    return out


def class_share_pct(in_dir: Path) -> dict:
    """{region: {month: % of domain cells in the class}}, mean over all seasons.

    From the per-month valid cell-hour tables of the same run. UTQ is one cell
    and is never dropped, so it gets 100.
    """
    norm_dir = in_dir.parent / "era5_hour_count_csv" / "normalization"
    tables = {}
    for name in CLASS_FILES:
        found = sorted(norm_dir.glob(f"ERA5_allValidCellHours_perMonth_*_{name}.csv"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one cell-hour table for {name} in {norm_dir}")
        # Total cell-hours of the class per month, summed over the seasons.
        tables[name] = pd.read_csv(found[0]).set_index("month").sum(axis=1)
    total = sum(tables.values())                       # all cells x all hours, per month
    out = {"UTQ": {m: 100.0 for m in MONTHS}}
    for region in REGIONS:
        if region != "UTQ":
            out[region] = {m: 100.0 * tables[region][calendar.month_abbr[m]]
                           / total[calendar.month_abbr[m]] for m in MONTHS}
    return out


def edge_color(color: str) -> tuple:
    """A darker shade of ``color`` for edges, whiskers and text (RGB x EDGE_DARKEN)."""
    r, g, b, _a = to_rgba(color)
    return (r * EDGE_DARKEN, g * EDGE_DARKEN, b * EDGE_DARKEN, 1.0)


def box_stats(values: np.ndarray, whis: str) -> dict | None:
    """Statistics for Axes.bxp. None when there are no values."""
    if values.size == 0:
        return None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    if whis == "minmax":
        lo, hi = values.min(), values.max()
    else:
        # Tukey: the furthest data points within 1.5 IQR of the box.
        iqr = q3 - q1
        lo = values[values >= q1 - TUKEY_WHIS * iqr].min()
        hi = values[values <= q3 + TUKEY_WHIS * iqr].max()
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi, "fliers": []}


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def plot(data: dict, hours: dict, share: dict, n_seasons: int, state: str, whis: str,
         region_label: str, path: Path) -> Path:
    """Four boxes per month, coloured by region."""
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    x = np.arange(len(MONTHS))
    n_reg = len(REGIONS)
    w = 0.18                                            # box width, axis units
    offsets = (np.arange(n_reg) - (n_reg - 1) / 2.0) * (w + 0.03)

    lo_all, hi_all, any_thin = np.inf, -np.inf, False
    for (region, (label, color)), off in zip(REGIONS.items(), offsets):
        # A class-month under MIN_CLASS_SHARE_PCT of the domain is not drawn.
        stats = [box_stats(data[region][m], whis)
                 if share[region][m] >= MIN_CLASS_SHARE_PCT else None for m in MONTHS]
        keep = [j for j, s in enumerate(stats) if s is not None]
        if not keep:
            continue
        bp = ax.bxp([stats[j] for j in keep], positions=x[keep] + off, widths=w,
                    showfliers=False, patch_artist=True, manage_ticks=False)
        for art, j in zip(bp["boxes"], keep):
            thin = data[region][MONTHS[j]].size < MIN_HOURS
            any_thin |= thin
            art.set_facecolor(to_rgba(color, BOX_ALPHA * (0.4 if thin else 1.0)))
            art.set_edgecolor(edge_color(color))
            art.set_linewidth(1.4)
            if thin:
                art.set_hatch("////")
        for key in ("whiskers", "caps"):
            for art in bp[key]:
                art.set_color(edge_color(color))
                art.set_linewidth(1.3)
        for art in bp["medians"]:
            art.set_color("black")
            art.set_linewidth(1.8)
        lo_all = min(lo_all, min(stats[j]["whislo"] for j in keep))
        hi_all = max(hi_all, max(stats[j]["whishi"] for j in keep))

    # Hours per month in the state for a typical grid cell, beneath each box.
    span = hi_all - lo_all
    y_txt = lo_all - 0.04 * span
    for (region, (_label, color)), off in zip(REGIONS.items(), offsets):
        for j, m in enumerate(MONTHS):
            if share[region][m] < MIN_CLASS_SHARE_PCT:
                # Say why the slot is empty, in the space the box would take.
                ax.text(x[j] + off, y_txt + 0.03 * span,
                        f"< {MIN_CLASS_SHARE_PCT:g}% of domain", rotation=90,
                        ha="center", va="bottom", fontsize=7.5, color=edge_color(color))
                ax.text(x[j] + off, y_txt, "—", ha="center", va="top", fontsize=8.5,
                        color=edge_color(color))
                continue
            n_h = hours[region][m]
            ax.text(x[j] + off, y_txt, f"{n_h:.0f}" if n_h > 0 else "—",
                    ha="center", va="top", fontsize=8.5, color=edge_color(color))

    # 0 deg C: the files carry no supercooled filter, so values above it are possible.
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1.0, zorder=0)
    ax.set_ylim(lo_all - 0.12 * span, max(hi_all, 0.0) + 0.20 * span)

    ax.set_ylabel(f"Cloud temperature, {STATES[state]} hours [°C]", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([calendar.month_abbr[m] for m in MONTHS], fontsize=12)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=to_rgba(c, BOX_ALPHA), ec=edge_color(c), lw=1.4)
               for _l, c in REGIONS.values()]
    labels = [lab for lab, _c in REGIONS.values()]
    handles.append(Line2D([0], [0], color="black", lw=1.8))
    labels.append("median")
    if any_thin:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=to_rgba("0.5", 0.18), ec="0.5",
                                     lw=1.4, hatch="////"))
        labels.append(f"fewer than {MIN_HOURS} h")
    whis_txt = ("whiskers: 1.5 IQR, no outliers drawn" if whis == "tukey"
                else "whiskers: min to max, no outliers drawn")
    ax.legend(handles, labels, ncol=len(handles), loc="upper right", framealpha=0.9,
              fontsize=10, title=f"{whis_txt};  beneath each box: mean hours per month for a typical grid cell",
              title_fontsize=9)
    ax.set_title(f"ERA5 cloud temperature by month and surface, {region_label}, "
                 f"{n_seasons} seasons (no supercooled filter)", loc="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=350, facecolor="white")
    plt.close(fig)
    return path


def print_summary(data: dict, hours: dict, share: dict, n_seasons: int) -> None:
    """Medians, hours per month (per cell, and values in the box), share at or above 0 deg C."""
    rows = []
    for region, (label, _c) in REGIONS.items():
        for m in MONTHS:
            v = data[region][m]
            rows.append({"region": label, "month": calendar.month_abbr[m],
                         "median_degC": np.median(v) if v.size else np.nan,
                         "hours_per_month_typical_cell": hours[region][m],
                         "pct_of_domain_cells": share[region][m],
                         "values_per_month_in_box": v.size / n_seasons,
                         "pct_at_or_above_0C": 100.0 * (v >= 0.0).mean() if v.size else np.nan})
    table = pd.DataFrame(rows).pivot(index="region", columns="month")
    for col in ("median_degC", "hours_per_month_typical_cell", "pct_of_domain_cells",
                "values_per_month_in_box",
                "pct_at_or_above_0C"):
        print(f"\n{col}")
        print(table[col][[calendar.month_abbr[m] for m in MONTHS]]
              .reindex([lab for lab, _c in REGIONS.values()]).round(2).to_string())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR,
                   help="folder with the ERA5_CloudTemp_*_perHour_*_<CLASS>.csv files")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where the figure goes (default: a 'figures' folder beside --in-dir)")
    p.add_argument("--state", choices=list(STATES), default="liquid_containing",
                   help="sky state to draw (default liquid_containing)")
    p.add_argument("--whis", choices=("tukey", "minmax"), default="tukey",
                   help="whisker reach: 1.5 IQR (default) or min to max")
    p.add_argument("--region-label", default=DEFAULT_REGION_LABEL,
                   help="domain name for the title")
    opts = p.parse_args(argv)

    data, n_seasons = load(opts.in_dir, opts.state)
    hours = per_cell_hours(opts.in_dir, opts.state)
    share = class_share_pct(opts.in_dir)
    out_dir = opts.out_dir or opts.in_dir.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    weighting = "_areaWeighted" if "areaWeighted" in find_file(opts.in_dir, "UTQ").name else ""
    suffix = "" if opts.whis == "tukey" else "_minmaxWhiskers"
    path = out_dir / f"ERA5_CloudTemp{weighting}_monthlyBoxes_{opts.state}{suffix}.png"
    print_summary(data, hours, share, n_seasons)
    dropped = [f"{REGIONS[r][0]} {calendar.month_abbr[m]}" for r in REGIONS for m in MONTHS
               if share[r][m] < MIN_CLASS_SHARE_PCT]
    print(f"\n  not drawn (< {MIN_CLASS_SHARE_PCT:g}% of domain): {', '.join(dropped) or 'none'}")
    print(f"  -> {plot(data, hours, share, n_seasons, opts.state, opts.whis, opts.region_label, path)}")


if __name__ == "__main__":
    main()
