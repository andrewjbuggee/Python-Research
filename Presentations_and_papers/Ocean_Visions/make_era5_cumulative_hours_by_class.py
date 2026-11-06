#!/usr/bin/env python3
"""Average cumulative cloud hours per grid cell through the season, by class.

WHAT IT DRAWS
=============
A two-panel figure from 1 October to 31 March, in the layout of
``plot_surface_class_timeseries.make_figure``:

    top      average cumulative hours per grid cell, one line per surface class
             plus the Utqiagvik (UTQ) cell
    bottom   area of the region held by each class (%), averaged over seasons

THE TOP PANEL
=============
For one class in one season, with t running over that season's hours:

    N(t) = valid grid cells in the class at hour t
    n(t) = those cells that pass the filter and hold the cloud type
    H(d) = sum over t up to the end of day d of n(t) / N(t)    (h per cell)

Hours with N(t) = 0 add nothing, so H stays flat while a class does not exist.
The plotted value on a calendar day is H averaged over all seasons, so on
31 March each line equals the mean seasonal total in
``precip_sensitivity/ERA5_avgHoursPerCell_perSeason_*_noPrecip.csv``. The script
checks this and stops if it fails.

The per-day sums come from ``make_era5_class_per_cell_csvs.daily_sums``, the
same function behind the per-season tables, and the counts come from
``make_era5_hour_count_csvs.count_cell_hours``, so the filter is the one used in
those tables: overcast (tcc >= 0.95), cloud water above the 0.01 g m-2 floors,
not precipitating (tp < 0.05 mm hr-1).

29 February exists in only some seasons. Days are aligned by calendar date, and
a season without that date adds zero hours to it, which is the only choice that
keeps the 31 March value equal to the mean seasonal total.

THE BOTTOM PANEL
================
The share of the region's area in each class, weighted by cos(latitude) as in
the original figure, averaged over each day's hours and then over seasons.

CELL WEIGHTING (choose with --cell-weighting)
=============================================
``area`` (default) counts each cell-hour in N(t) and n(t) as cos(latitude), so
the top panel is per unit area of the class. ``uniform`` counts each as 1, so a
grid cell is a grid cell. Each mode checks against the season tables of the same
weighting. A separate recount put the difference for pack ice's seasonal total
at about 1% over the Barrow domain (1588 vs 1608 h); it can be larger over a
domain with a wider latitude range.

READING THE LINES
=================
A class's slope is how often its cells held the cloud type. A flat line means
either that the cloud type was absent or that the class itself was absent, and
the bottom panel tells which. Open ocean's line flattens from December because
the class disappears, not because its clouds change.

Usage
-----
    python make_era5_cumulative_hours_by_class.py                 # ~5 min
    python make_era5_cumulative_hours_by_class.py --plot-only     # redraw from CSVs
    python make_era5_cumulative_hours_by_class.py --plot-only --cloud-type ice_only
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ov_config as ov  # first: thread limits and sys.path on Casper

# The counting pass and the per-hour normalisation are reused, not copied, so
# these lines cannot disagree with the per-season tables.
from make_era5_class_per_cell_csvs import daily_sums
from make_era5_hour_count_csvs import (
    FILE_TAG,
    OUT_PRODUCT,
    PHASE_CATS,
    SETTINGS,
    SLOT_CLASS,
    active_slots,
    build_args,
    count_cell_hours,
    git_describe,
    season_label,
)

import plot_lwp_histogram_by_surface_class as lwph  # noqa: E402
from plot_surface_class_timeseries import SITE_COLOR, month_ticks  # noqa: E402
from seb_analysis_common import (  # noqa: E402
    load_seb_data,
    resolve_region_dir,
    season_calendar,
)
from surface_classification import (  # noqa: E402
    CLASS_CODES,
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_ORDER,
    align_lsm_to_grid,
    area_weights_2d,
    classify_cells,
    iter_time_blocks,
    load_land_sea_mask,
)

UTQ = "UTQ"
CLOUD_TYPE_LABELS = {
    "liquid_containing": "liquid-containing",
    "liquid_only": "liquid-only",
    "ice_only": "ice-only",
}
# File column name of each module class, in the stacking order of the original
# figure (bottom of the stack first).
COLUMN_OF_CLASS = {cls: column for column, cls in SLOT_CLASS.items()}
AREA_COLUMNS = [COLUMN_OF_CLASS[cls] for cls in CLASS_ORDER]
# "Pack ice" rather than the module's "Sea ice", as in the tables and in Genie's
# figure; the other class labels are the module's own.
LINE_LABELS = {**CLASS_LABELS, "sea_ice": "Pack ice"}


# ----------------------------------------------------------------------------
# Calendar: one row per (month, day) of a leap-year season
# ----------------------------------------------------------------------------
def slots_of_season(args: argparse.Namespace) -> list[tuple[int, int]]:
    """Ordered (month, day) from season start to end, including 29 February."""
    return season_calendar(args.season_start, args.season_end)


def slot_labels(slots: list[tuple[int, int]]) -> list[str]:
    """'1 Oct' style row labels for the CSV files."""
    return [f"{d} {pd.Timestamp(2000, m, 1):%b}" for m, d in slots]


def spread_to_seasons(values: np.ndarray, C: dict, slots: list) -> np.ndarray:
    """Move a per-day array to (season, calendar slot, ...), NaN on missing dates.

    ``values`` has the day axis first, in the order of ``C["dates"]``. Only
    29 February of a common year stays NaN.
    """
    dates = pd.to_datetime(C["dates"])
    slot_of = {md: i for i, md in enumerate(slots)}
    season_of = {y: i for i, y in enumerate(C["seasons"])}
    out = np.full((len(C["seasons"]), len(slots), *values.shape[1:]), np.nan)
    for day, date in enumerate(dates):
        out[season_of[int(C["day_season"][day])], slot_of[(date.month, date.day)]] = values[day]
    return out


# ----------------------------------------------------------------------------
# Top panel: average cumulative hours per cell
# ----------------------------------------------------------------------------
def cumulative_hours(C: dict, slots: list) -> dict[str, pd.DataFrame]:
    """Mean-over-seasons cumulative hours per cell, one table per cloud type.

    Each table has rows = calendar days and columns = UTQ (if in the domain) and
    the classes. Units are hours per grid cell (plain hours for UTQ).
    """
    sum_f = daily_sums(C)["sum_f"]                          # (day, slot, type), h per day
    per_season = spread_to_seasons(sum_f, C, slots)         # (season, day, slot, type)
    # A date missing from a season adds nothing to the running total. NaN would
    # otherwise propagate through every later day of that season.
    cum = np.cumsum(np.nan_to_num(per_season), axis=1)      # (season, day, slot, type)
    mean = cum.mean(axis=0)                                 # (day, slot, type)

    slot_ids, names = zip(*active_slots(C))
    labels = slot_labels(slots)
    return {
        cat: pd.DataFrame(mean[:, list(slot_ids), k], index=labels, columns=list(names))
        for k, cat in enumerate(PHASE_CATS)
    }


# ----------------------------------------------------------------------------
# Bottom panel: area share of each class
# ----------------------------------------------------------------------------
def class_area_percent(C: dict, args: argparse.Namespace, slots: list) -> pd.DataFrame:
    """cos(lat)-weighted area share of each class (%), mean over seasons, per day.

    The counting pass keeps plain cell counts only, so this is one further pass
    that reads sea ice concentration alone, through the same classifier.
    """
    ds = load_seb_data(args.region, None, None, resolve_region_dir(args).parent)
    lsm = align_lsm_to_grid(
        load_land_sea_mask(
            args.region, lwph.resolve_data_root(args.storage, args.data_root), args.mask_grid
        ),
        ds,
    )
    weights = area_weights_2d(ds["latitude"].values, ds.sizes["longitude"])  # (lat, lon)

    # Only the hours the counting pass used, assigned to its day axis.
    times = np.asarray(ds["valid_time"].values)
    keep = np.isin(times, C["hour_time"])
    day_of_step = np.searchsorted(C["dates"], times.astype("datetime64[D]"))

    n_day = C["dates"].size
    weight_day = np.zeros((n_day, len(CLASS_ORDER)))        # (day, class), summed cos(lat)
    hours_day = np.zeros(n_day)                             # (day,), hours read

    print("  Reading sea ice concentration for the area panel ...", flush=True)
    for i0, block in iter_time_blocks(ds, ["siconc"], args.block_hours, keep_mask=keep):
        span = slice(i0, i0 + block.sizes["valid_time"])
        sel = keep[span]
        if not sel.any():
            continue
        classes = classify_cells(
            lsm,
            block["siconc"].values,
            args.lsm_tol,
            args.open_ocean_max_siconc,
            args.sea_ice_min_siconc,
            args.land_max_siconc,
        )[sel]                                              # (t, lat, lon)
        day = day_of_step[span][sel]                        # (t,)
        np.add.at(hours_day, day, 1.0)
        for k, name in enumerate(CLASS_ORDER):
            # Summed area weight of this class at each hour, added to its day.
            area_t = ((classes == CLASS_CODES[name]) * weights).sum(axis=(1, 2))
            np.add.at(weight_day[:, k], day, area_t)

    pct_day = 100.0 * weight_day / (hours_day[:, None] * weights.sum())  # (day, class), %
    per_season = spread_to_seasons(pct_day, C, slots)       # (season, day, class)
    # nanmean: 29 February is averaged over the leap seasons only.
    mean = np.nanmean(per_season, axis=0)                   # (day, class)
    return pd.DataFrame(mean, index=slot_labels(slots), columns=AREA_COLUMNS)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_area(area_pct: pd.DataFrame) -> None:
    """The classes partition the domain, so each day must sum to 100%."""
    worst_pct = float((area_pct.sum(axis=1) - 100.0).abs().max())
    if worst_pct > 1.0e-6:
        raise AssertionError(f"class areas miss 100% by {worst_pct:.2e} percentage points")
    print(f"  area check OK: classes sum to 100% (worst miss {worst_pct:.1e} points)")


def check_against_season_tables(
    cum: dict, span: str, hour_count_dir: Path, weighting: str
) -> None:
    """The 31 March values must equal the mean seasonal totals already written."""
    tol_h = 1.0e-3  # h; those tables are written to 4 decimals
    for cat in PHASE_CATS:
        name = ov.weighted_names(
            f"ERA5_avgHoursPerCell_perSeason_{span}_{FILE_TAG[cat]}_noPrecip.csv", weighting
        )
        path = hour_count_dir / f"precip_sensitivity{ov.weight_suffix(weighting)}" / name
        if not path.exists():
            print(f"  !! no table to check against: {path.name}")
            continue
        ref = pd.read_csv(path, index_col="season").loc["mean_of_seasons"]
        final = cum[cat].iloc[-1]
        worst_h = float((final - ref[final.index]).abs().max())
        if worst_h > tol_h:
            raise AssertionError(f"{cat}: 31 March differs from {name} by {worst_h:.5f} h")
        print(f"  {cat}: 31 March matches the seasonal table (worst {worst_h:.5f} h)")


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def season_words(span: str) -> str:
    """'2014_2025' -> '11 seasons, 2014/15 to 2024/25', from a span tag."""
    parts = span.split("_")
    first, last = int(parts[0]), int(parts[1]) - 1
    # A non-contiguous selection states its own count; see ov_config.span_tag.
    count = int(parts[2].replace("seasons", "")) if len(parts) > 2 else last - first + 1
    return f"{count} seasons, {season_label(first)} to {season_label(last)}"


def plot_figure(
    out_dir: Path, span: str, cat: str, args: argparse.Namespace, region: str
) -> Path:
    """Draw the figure from the CSVs, so it cannot disagree with them."""
    # Agg because the script also runs on Casper compute nodes with no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weighting = args.cell_weighting
    cum = pd.read_csv(
        out_dir / ov.weighted_names(
            f"ERA5_cumulativeHoursPerCell_perDay_{span}_{FILE_TAG[cat]}.csv", weighting
        ),
        index_col="date",
    )
    area_pct = pd.read_csv(out_dir / f"ERA5_classAreaPercent_perDay_{span}.csv", index_col="date")
    slots = slots_of_season(args)
    x = np.arange(len(slots))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )

    # --- top: one cumulative line per class -----------------------------------
    # Each legend entry carries the 31 March total, because lines that finish
    # close together (pack ice and UTQ) cannot be read apart from the plot.
    for name in CLASS_ORDER:
        column = COLUMN_OF_CLASS[name]
        ax_top.plot(
            x, cum[column], color=CLASS_COLORS[name], linewidth=1.8, solid_capstyle="round",
            label=f"{LINE_LABELS[name]} ({cum[column].iloc[-1]:.0f} h)",
        )
    # Last, dashed and black, as in the original: a point comparison, not a class.
    if UTQ in cum.columns:
        ax_top.plot(
            x, cum[UTQ], color=SITE_COLOR, linewidth=1.7, linestyle="--", zorder=5,
            label=f"Utqiagvik (ARM site) ({cum[UTQ].iloc[-1]:.0f} h)",
        )
    ax_top.set_ylabel("Average cumulative hours, area weighted [h]" if weighting == "area"
                      else "Average cumulative hours per grid cell [h]")
    ax_top.set_ylim(0, None)
    ax_top.grid(alpha=0.25, linewidth=0.6)
    # Upper left, unlike the original: rising lines leave that corner empty.
    ax_top.legend(
        loc="upper left", ncol=2, frameon=True, framealpha=0.9, fontsize=10,
        columnspacing=1.1, handlelength=1.6,
    )

    # --- bottom: area share, the same stack as the original --------------------
    ax_bot.stackplot(
        x, *[area_pct[COLUMN_OF_CLASS[name]].values for name in CLASS_ORDER],
        colors=[CLASS_COLORS[name] for name in CLASS_ORDER], edgecolor="none",
    )
    ax_bot.set_ylim(0, 100)
    ax_bot.set_ylabel("Area of region [%]")
    ax_bot.set_xlabel(
        f"Day of season ({args.season_start[0]:02d}-{args.season_start[1]:02d} "
        f"to {args.season_end[0]:02d}-{args.season_end[1]:02d})"
    )
    ax_bot.grid(alpha=0.25, linewidth=0.6, color="white")

    pos, lab = month_ticks(slots)
    ax_bot.set_xticks(pos)
    ax_bot.set_xticklabels(lab)
    ax_bot.set_xlim(0, len(slots) - 1)

    fig.suptitle(
        f"Cumulative {CLOUD_TYPE_LABELS[cat]} cloud hours by surface class — {region}\n"
        f"mean of {season_words(span)}   |   each hour divided by that hour's cell count   |   "
        f"tcc $\\geq$ {SETTINGS['min_cloud_fraction']:g}, "
        f"tp < {SETTINGS['precip_rate_max']:g} mm hr$^{{-1}}$",
        fontsize=13, y=0.965,
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.075, right=0.985)
    path = out_dir / ov.weighted_names(
        f"ERA5_cumulativeHoursPerCell_{span}_{FILE_TAG[cat]}.png", weighting
    )
    fig.savefig(path, dpi=350, facecolor="white")   # 350 dpi, as the other OV figures
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------
def write_tables(
    cum: dict, area_pct: pd.DataFrame, out_dir: Path, span: str, weighting: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat, df in cum.items():
        df.to_csv(
            out_dir / ov.weighted_names(
                f"ERA5_cumulativeHoursPerCell_perDay_{span}_{FILE_TAG[cat]}.csv", weighting
            ),
            index_label="date", float_format="%.4f",
        )
    area_pct.to_csv(
        out_dir / f"ERA5_classAreaPercent_perDay_{span}.csv",
        index_label="date", float_format="%.4f",
    )
    area = weighting == "area"
    unit_words = ("hours, each cell-hour counted as cos(latitude)" if area
                  else "hours per grid cell")
    weight_sentence = ("Cumulative hours above are also area weighted." if area
                       else "Cumulative hours above are NOT area weighted.")
    text = f"""AVERAGE CUMULATIVE CLOUD HOURS PER GRID CELL, BY SURFACE CLASS
==============================================================

Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_cumulative_hours_by_class.py
(repo {git_describe()}).

Data, filters, surface classes and cloud types are as in ../readME.txt.

  ERA5_cumulativeHoursPerCell_perDay_{span}_<TYPE>.csv
      rows = calendar days, 1 Oct to 31 Mar including 29 Feb; columns = classes.
      For each season: the running sum of n(t)/N(t) to the end of that day
      ({unit_words}), then the mean over seasons. A class's hours with
      no cells, and 29 Feb in common years, add nothing. The 31 Mar row equals
      the mean row of ../precip_sensitivity{ov.weight_suffix(weighting)}/ERA5_avgHoursPerCell_perSeason_*_noPrecip.csv.
      <TYPE> = liquidContaining | liquidOnly | iceOnly.

  ERA5_classAreaPercent_perDay_{span}.csv
      area of the region held by each class (%), cos(latitude) weighted, mean
      over seasons. {weight_sentence}

  ERA5_cumulativeHoursPerCell_{span}_<TYPE>.png
      the figure drawn from these two tables.
"""
    (out_dir / "readME.txt").write_text(ov.weighted_names(text, weighting))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def find_span(out_dir: Path) -> str:
    """The season span tag of the area table already in ``out_dir``."""
    prefix, suffix = "ERA5_classAreaPercent_perDay_", ".csv"
    found = sorted(out_dir.glob(f"{prefix}*{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one {prefix}<span>{suffix} in {out_dir}, found {len(found)}; "
            "run without --plot-only first"
        )
    return found[0].name[len(prefix) : -len(suffix)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory (default <hour-count folder>/cumulative_by_class)",
    )
    p.add_argument(
        "--cloud-type", choices=PHASE_CATS, default="liquid_containing",
        help="cloud type to draw (default liquid_containing); all three go to CSV",
    )
    p.add_argument(
        "--plot-only", action="store_true",
        help="redraw the figure from the CSVs in the output folder, no reading of ERA5",
    )
    ov.add_weighting_arg(p)
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    hour_count_dir = run.out_dir(OUT_PRODUCT)
    weighting = opts.cell_weighting
    out_dir = opts.out_dir or hour_count_dir / f"cumulative_by_class{ov.weight_suffix(weighting)}"
    args = build_args({**SETTINGS, **run.settings(), "cell_weighting": weighting})

    if opts.plot_only:
        span = find_span(out_dir)
        print(f"  -> {plot_figure(out_dir, span, opts.cloud_type, args, run.domain.name)}")
        return

    print("=" * 72)
    print("ERA5 average cumulative cloud hours per grid cell, by surface class")
    print(run.report())
    print("=" * 72)
    C = count_cell_hours(args)
    span = ov.span_tag(C["seasons"])
    slots = slots_of_season(args)

    cum = cumulative_hours(C, slots)
    area_pct = class_area_percent(C, args, slots)
    check_area(area_pct)
    check_against_season_tables(cum, span, hour_count_dir, weighting)

    write_tables(cum, area_pct, out_dir, span, weighting)
    figure = plot_figure(out_dir, span, opts.cloud_type, args, run.domain.name)
    print(f"\n  31 March, {opts.cloud_type} (h per cell):")
    print(cum[opts.cloud_type].iloc[-1].round(1).to_string())
    print(f"\n  -> CSV files + readME.txt + {figure.name} in {out_dir}")


if __name__ == "__main__":
    main()
