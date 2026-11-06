#!/usr/bin/env python3
"""How much does the no-precipitation filter change the per-cell cloud hours?

WHAT IT COMPUTES
================
Average hours per grid cell, per season, for UTQ and each of the five surface
classes, under two filters that differ ONLY in the precipitation test:

    as is            overcast (tcc >= 0.95), cloud water above the floors, and
                     not precipitating (tp < 0.05 mm hr-1)
    precip allowed   the same, with the precipitation test removed

and, from the pair, the percentage of data lost to the precipitation test.

HOW THE "ONLY ONE DIFFERENCE" IS GUARANTEED
===========================================
The counting pass is ``make_era5_hour_count_csvs.count_cell_hours``, unchanged,
called twice. An hour counts as raining when ``tp >= args.precip_rate_max``
(the only use of that setting in the pass). Setting the threshold to infinity
makes the test false for every hour, so there is no second copy of the filters
that could drift from the first. Two passes take about 8 minutes in total.

THE AVERAGE
===========
The ``per_hour`` method of ``make_era5_class_per_cell_csvs.py``, for class c:

    avg hours per cell = sum over t of n(t) / N(t),  over hours with N(t) > 0

    N(t) = valid cells of the class at hour t
    n(t) = those cells passing the filter and holding the cloud type

UTQ is one cell, so its value is plain hours. The "as is" liquid-containing
values are the SLC bars of the seasonal-frequency figure in Genie's paper.

THE PERCENTAGE LOST
===================
    % lost = 100 * (allowed - as is) / allowed

This is the share of the filtered-for-everything-else population that the
precipitation test removes. Two versions are written, because "data" can mean
either, and they differ for classes whose size changes:

    percentHoursPerCellLost   from the avg hours per cell above
    percentCellHoursLost      from raw (cell, hour) counts, so a cell-hour in a
                              large pack ice hour weighs the same as one in a
                              small pack ice hour

Each table ends in two summary rows, which also differ slightly:

    mean_of_seasons    arithmetic mean of the seasonal percentages
    pooled_all_seasons 100 * (sum allowed - sum as is) / sum allowed

CAUTION ON READING THE PRECIP-ALLOWED NUMBERS
=============================================
Phase comes from tclw and tciw only. ERA5 carries falling snow and rain as
separate species, so frozen mass that has become snow does not count towards
IWP, and a snowing column can look more liquid-rich than its total condensate
would make it. This is why the precip-allowed liquid-containing numbers should
not be read as an unbiased count; the effect has not been quantified here.

Usage
-----
    python make_era5_precip_filter_sensitivity.py
    python make_era5_precip_filter_sensitivity.py --out-dir /path/to/dir
    python make_era5_precip_filter_sensitivity.py --plot-only   # redraw from the CSVs

THE FIGURE
==========
A full run also draws the liquid-containing (SLC) comparison beside the tables:
one panel per surface class, paired bars per season, one shared y-axis.
``--plot-only`` redraws it from the CSVs in seconds, with no counting passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

# First, before numpy: on Casper it sets thread limits and the cache location,
# and it puts the analysis modules on sys.path.
import ov_config as ov

import numpy as np
import pandas as pd

# The counting pass, the figure settings and the per_hour average all come from
# the existing scripts, so these tables share their population exactly.
from make_era5_class_per_cell_csvs import daily_sums, finish
from make_era5_hour_count_csvs import (
    CATS,
    OUT_PRODUCT,
    FILE_TAG,
    I_UTQ,
    PHASE_CATS,
    SETTINGS,
    SLOT_NAMES,
    build_args,
    count_cell_hours,
    git_describe,
    season_label,
)

# Infinity disables the test: no finite tp is >= inf, so no hour is "raining".
PRECIP_ALLOWED = {"precip_rate_max": np.inf}

# The classes drawn in Genie's figure (UTQ is shown there too, but is a site,
# not a surface class). Coastal and MarginalSeaIce are not drawn.
CLASSES_IN_FIGURE = ("Land", "OpenOcean", "PackIce")
ALL_CLASSES = tuple(name for name in SLOT_NAMES if name != "UTQ")

I_ALL_VALID = CATS.index("all_valid")
ROW_MEAN = "mean_of_seasons"
ROW_POOLED = "pooled_all_seasons"

# Figure styling. The two filters are the only thing colour encodes, and the
# panels name the class, so the same pair is used in every panel. Purple and
# teal avoid the colours the Ocean Visions figures already assign to cloud
# phase and to the surface classes. Checked for colour-blind separation with
# OKLab distance x 100 after Machado et al. (2009) simulation at full severity:
# normal 30, protan 30, deutan 25, tritan 25 (a target of >= 8 is common).
COLOR_AS_IS = "#5e3c99"
COLOR_PRECIP_ALLOWED = "#2ca58d"
INK_PRIMARY = "#222222"    # titles and axis labels
INK_MUTED = "#666666"      # tick labels and panel summaries
GRID_COLOR = "#e4e4e4"     # recessive gridlines
PANEL_LABELS = {
    "UTQ": "UTQ grid cell",
    "Land": "Land",
    "Coastal": "Coastal",
    "OpenOcean": "Open ocean",
    "MarginalSeaIce": "Marginal ice zone",
    "PackIce": "Pack ice",
}


# ----------------------------------------------------------------------------
# Seasonal tables: rows = seasons, columns = UTQ + the five classes
# ----------------------------------------------------------------------------
def drop_absent_site(df: pd.DataFrame, C: dict) -> pd.DataFrame:
    """Remove the UTQ column when the ARM site is outside the domain."""
    return df if C["has_site"] else df.drop(columns=SLOT_NAMES[I_UTQ])


def hours_per_cell(C: dict) -> dict[str, pd.DataFrame]:
    """Average hours per cell per season, one table per cloud type (h)."""
    D = daily_sums(C)
    labels = [season_label(y) for y in C["seasons"]]
    # (n_season, n_slot, n_type): finish() returns the per_hour average per type.
    values = np.array(
        [
            [
                finish(D, C["day_season"] == y, slot, "per_hour")[0]
                for slot in range(len(SLOT_NAMES))
            ]
            for y in C["seasons"]
        ]
    )
    return {
        cat: drop_absent_site(
            pd.DataFrame(values[:, :, k], index=labels, columns=SLOT_NAMES), C
        )
        for k, cat in enumerate(PHASE_CATS)
    }


def cell_hours(C: dict) -> dict[str, pd.DataFrame]:
    """(cell, hour) totals per season, one table per cloud type.

    Plain counts under uniform weighting; each cell-hour counted as cos(latitude)
    under area weighting, so the percentage lost is a share of cloud area-time.
    """
    labels = [season_label(y) for y in C["seasons"]]
    out = {}
    for cat in PHASE_CATS:
        ci = CATS.index(cat)
        # counts is (day, slot, category); summing over a season's days leaves
        # one count per slot.
        if C["hourly_w"] is None:
            rows = [C["counts"][C["day_season"] == y, :, ci].sum(axis=0) for y in C["seasons"]]
        else:
            season_of_hour = C["day_season"][C["hour_day"]]        # (hour,)
            rows = [C["hourly_w"][season_of_hour == y, :, ci].sum(axis=0) for y in C["seasons"]]
        out[cat] = drop_absent_site(
            pd.DataFrame(np.array(rows, dtype=float), index=labels, columns=SLOT_NAMES), C
        )
    return out


def percent_lost(as_is: pd.DataFrame, allowed: pd.DataFrame) -> pd.DataFrame:
    """100 * (allowed - as is) / allowed, per season, plus the two summary rows."""
    pct = 100.0 * (allowed - as_is) / allowed.where(allowed > 0)
    pct.loc[ROW_MEAN] = pct.mean(axis=0)
    # Pooled is computed from the sums, not from the percentages above, so the
    # seasonal rows must be the only ones summed.
    total_allowed = allowed.sum(axis=0)
    pct.loc[ROW_POOLED] = 100.0 * (total_allowed - as_is.sum(axis=0)) / total_allowed.where(
        total_allowed > 0
    )
    return pct


def with_mean_row(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of a seasonal table with its across-season mean appended."""
    out = df.copy()
    out.loc[ROW_MEAN] = df.mean(axis=0)
    return out


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_pair(C_as_is: dict, C_allowed: dict) -> None:
    """Things that must hold if the two passes differ only by the precip test."""
    # The denominator is taken before any cloud filter, so it must not move.
    if not np.array_equal(
        C_as_is["hourly"][:, :, I_ALL_VALID], C_allowed["hourly"][:, :, I_ALL_VALID]
    ):
        raise AssertionError("N(t) differs between the two passes")
    # Removing a test can only add cell-hours, for every hour, slot and type.
    idx = [CATS.index(c) for c in PHASE_CATS]
    if (C_allowed["hourly"][:, :, idx] < C_as_is["hourly"][:, :, idx]).any():
        raise AssertionError("allowing precipitation lost a cell-hour")
    print("  pair checks OK: N(t) identical; precip-allowed counts >= as-is everywhere")


def check_against_existing_csvs(
    as_is: dict[str, pd.DataFrame], span: str, hour_count_dir: Path, weighting: str
) -> None:
    """The as-is tables must reproduce the CSVs already delivered (rounded to 4 dp)."""
    tol_h = 1.0e-4  # h; those files are written to 4 decimals
    worst_h = 0.0
    compared = 0
    for name in as_is[PHASE_CATS[0]].columns:
        # UTQ is a single cell, so its per-cell table is the cell-hour table.
        path = (
            hour_count_dir / f"ERA5_totalHours_perSeason_{span}_UTQ.csv"
            if name == "UTQ"
            else hour_count_dir
            / f"class_normalized{ov.weight_suffix(weighting)}"
            / ov.weighted_names(f"ERA5_avgHoursPerCell_perSeason_{span}_{name}.csv", weighting)
        )
        if not path.exists():
            print(f"  !! no existing table to compare for {name}: {path.name}")
            continue
        ref = pd.read_csv(path).set_index("season")
        for cat in PHASE_CATS:
            worst_h = max(worst_h, float((as_is[cat][name] - ref[cat]).abs().max()))
            compared += 1
    if worst_h > tol_h:
        raise AssertionError(f"as-is tables differ from existing CSVs by {worst_h:.5f} h")
    print(f"  as-is reproduces {compared} existing columns; worst difference {worst_h:.5f} h")


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
def print_tables(as_is, allowed, lost_hours, lost_cell_hours) -> None:
    """The liquid-containing (SLC) tables, plus every type's summary."""
    pd.set_option("display.width", 200)
    cat = "liquid_containing"
    print(f"\n  {cat}: average hours per cell, AS IS (no precipitation)")
    print(with_mean_row(as_is[cat]).round(0).to_string())
    print(f"\n  {cat}: average hours per cell, PRECIPITATION ALLOWED")
    print(with_mean_row(allowed[cat]).round(0).to_string())
    print(f"\n  {cat}: % of hours per cell lost to the precipitation test")
    print(lost_hours[cat].round(1).to_string())
    print(f"\n  {cat}: % of raw cell-hours lost to the precipitation test")
    print(lost_cell_hours[cat].round(1).to_string())
    for other in PHASE_CATS[1:]:
        print(f"\n  {other}: % of hours per cell lost, summary rows")
        print(lost_hours[other].loc[[ROW_MEAN, ROW_POOLED]].round(1).to_string())


def print_rankings(as_is: dict, allowed: dict) -> None:
    """Which class leads each season, under each filter, for SLC."""
    cat = "liquid_containing"
    for title, classes in (
        ("classes drawn in the figure", CLASSES_IN_FIGURE),
        ("all five classes", ALL_CLASSES),
    ):
        a = as_is[cat][list(classes)].idxmax(axis=1)
        b = allowed[cat][list(classes)].idxmax(axis=1)
        print(f"\n  {cat}: top class per season, {title}")
        print(pd.DataFrame({"as_is": a, "precip_allowed": b}).to_string())
        print(f"    seasons led, as is:          {a.value_counts().to_dict()}")
        print(f"    seasons led, precip allowed: {b.value_counts().to_dict()}")
        print(f"    seasons where the leader changes: {int((a != b).sum())}")


# ----------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------
def write_readme(out_dir: Path, span: str, has_site: bool, weighting: str) -> Path:
    utq_column = "UTQ, " if has_site else ""
    raw_word = "totals, each cell-hour counted as cos(latitude)" if weighting == "area" else "counts"
    text = f"""EFFECT OF THE NO-PRECIPITATION FILTER ON PER-CELL CLOUD HOURS
=============================================================

Generated {datetime.now(timezone.utc):%Y-%m-%d} by make_era5_precip_filter_sensitivity.py
(repo {git_describe()}).

Data, surface classes and cloud types are as in ../readME.txt. The average is
the per_hour method in ../class_normalized/readME_classNormalized.txt.

Two filters, differing only in the precipitation test:

  noPrecip       overcast (tcc >= 0.95), cloud water above the 0.01 g m-2
                 floors, and tp < 0.05 mm hr-1. These are the existing tables.
  precipAllowed  the same, with the tp test removed.

Every file has rows = seasons and columns = {utq_column}Land, Coastal, OpenOcean,
MarginalSeaIce, PackIce. <TYPE> = liquidContaining | liquidOnly | iceOnly.

  ERA5_avgHoursPerCell_perSeason_{span}_<TYPE>_noPrecip.csv
  ERA5_avgHoursPerCell_perSeason_{span}_<TYPE>_precipAllowed.csv
      average hours per cell (h); last row = mean of the seasons

  ERA5_percentHoursPerCellLost_perSeason_{span}_<TYPE>.csv
      100 * (precipAllowed - noPrecip) / precipAllowed, from the tables above

  ERA5_percentCellHoursLost_perSeason_{span}_<TYPE>.csv
      the same from raw (cell, hour) {raw_word}

  ERA5_avgHoursPerCell_perSeason_{span}_liquidContaining_precipComparison.png
      paired bars of the two liquid-containing tables, one panel per class

The percentage tables end in two summary rows:

  {ROW_MEAN}     arithmetic mean of the seasonal percentages
  {ROW_POOLED}  computed from the sums over all seasons

Phase uses tclw and tciw only, and ERA5 holds falling snow and rain as separate
species, so precipitating columns may read as more liquid-rich than their total
condensate. This effect has not been quantified.
"""
    path = out_dir / "readME.txt"
    path.write_text(ov.weighted_names(text, weighting))
    return path


def write_all(as_is, allowed, lost_hours, lost_cell_hours, out_dir: Path, span: str,
              weighting: str) -> int:
    """Every CSV; returns how many were written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    def save(df: pd.DataFrame, name: str, fmt: str) -> None:
        nonlocal n_written
        df.to_csv(out_dir / ov.weighted_names(name, weighting), index_label="season",
                  float_format=fmt)
        n_written += 1

    for cat in PHASE_CATS:
        tag = FILE_TAG[cat]
        save(
            with_mean_row(as_is[cat]),
            f"ERA5_avgHoursPerCell_perSeason_{span}_{tag}_noPrecip.csv",
            "%.4f",
        )
        save(
            with_mean_row(allowed[cat]),
            f"ERA5_avgHoursPerCell_perSeason_{span}_{tag}_precipAllowed.csv",
            "%.4f",
        )
        save(lost_hours[cat], f"ERA5_percentHoursPerCellLost_perSeason_{span}_{tag}.csv", "%.2f")
        save(lost_cell_hours[cat], f"ERA5_percentCellHoursLost_perSeason_{span}_{tag}.csv", "%.2f")
    return n_written


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def find_span(out_dir: Path, weighting: str) -> str:
    """The season span tag of the SLC tables already in ``out_dir``."""
    prefix = ov.weighted_names("ERA5_avgHoursPerCell_perSeason_", weighting)
    suffix = "_liquidContaining_noPrecip.csv"
    found = sorted(out_dir.glob(f"{prefix}*{suffix}"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one {prefix}<span>{suffix} in {out_dir}, found {len(found)}; "
            "run without --plot-only first"
        )
    return found[0].name[len(prefix) : -len(suffix)]


def plot_comparison(out_dir: Path, span: str, weighting: str,
                    cat: str = "liquid_containing") -> Path:
    """Paired seasonal bars, as is vs precipitation allowed, one panel per class.

    Reads the CSVs rather than taking arrays, so a full run and ``--plot-only``
    draw from the same numbers and the figure can never disagree with the tables.
    """
    # Imported here because only this function needs pyplot, and Agg because the
    # script also runs on Casper compute nodes with no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tag = FILE_TAG[cat]
    stem = ov.weighted_names(f"ERA5_avgHoursPerCell_perSeason_{span}_{tag}", weighting)
    as_is_h = pd.read_csv(out_dir / f"{stem}_noPrecip.csv", index_col="season")
    allowed_h = pd.read_csv(out_dir / f"{stem}_precipAllowed.csv", index_col="season")
    lost_pct = pd.read_csv(
        out_dir / ov.weighted_names(f"ERA5_percentHoursPerCellLost_perSeason_{span}_{tag}.csv",
                                    weighting),
        index_col="season",
    )
    # The seasonal rows become bars; the summary row becomes each panel's label.
    seasons = [s for s in as_is_h.index if s != ROW_MEAN]
    x = np.arange(len(seasons))
    bar_width = 0.37   # of one season's slot; the pair leaves a sliver of surface between

    classes = list(as_is_h.columns)
    n_rows = int(np.ceil(len(classes) / 2))
    # One shared y-axis, so bar heights compare directly between classes.
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(11.0, 2.75 * n_rows + 1.1), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    for i, (ax, name) in enumerate(zip(axes.ravel(), classes)):
        ax.bar(x - 0.2, as_is_h.loc[seasons, name], bar_width, color=COLOR_AS_IS)
        ax.bar(x + 0.2, allowed_h.loc[seasons, name], bar_width, color=COLOR_PRECIP_ALLOWED)
        ax.set_title(PANEL_LABELS.get(name, name), loc="left", fontsize=12,
                     fontweight="bold", color=INK_PRIMARY)
        # The across-season result, stated so it need not be estimated from bars.
        ax.set_title(
            f"mean {as_is_h.loc[ROW_MEAN, name]:.0f} \u2192 "
            f"{allowed_h.loc[ROW_MEAN, name]:.0f} h "
            f"({lost_pct.loc[ROW_MEAN, name]:.1f}% lost to filter)",
            loc="right", fontsize=9.5, color=INK_MUTED,
        )
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
        ax.tick_params(colors=INK_MUTED, length=0)
        ax.set_xticks(x, seasons, rotation=45, ha="right", fontsize=9)
        # Season labels on the last two panels only. With an odd panel count
        # (no UTQ) one of those sits above the blank slot, not on the bottom row.
        ax.tick_params(labelbottom=i >= len(classes) - 2)

    for ax in axes.ravel()[len(classes) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Area-weighted hours per season" if weighting == "area"
                      else "Hours per grid cell per season", fontsize=10, color=INK_PRIMARY)

    fig.suptitle(
        "ERA5 liquid-containing cloud hours with and without the no-precipitation filter",
        fontsize=14, fontweight="bold", color=INK_PRIMARY, x=0.5, y=0.985,
    )
    fig.legend(
        handles=[
            Patch(color=COLOR_AS_IS, label=f"As is: tp < {SETTINGS['precip_rate_max']:g} mm hr$^{{-1}}$"),
            Patch(color=COLOR_PRECIP_ALLOWED, label="Precipitation allowed"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False,
        fontsize=11, labelcolor=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out_dir / f"{stem}_precipComparison.png"
    fig.savefig(path, dpi=350, facecolor="white")   # 350 dpi, as the other OV figures
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default <hour-count folder>/precip_sensitivity)",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figure from the CSVs already in the output folder, no counting",
    )
    ov.add_weighting_arg(p)
    ov.add_run_args(p)
    opts = p.parse_args(argv)
    run = ov.run_from_args(opts)
    hour_count_dir = run.out_dir(OUT_PRODUCT)
    weighting = opts.cell_weighting
    out_dir = opts.out_dir or hour_count_dir / f"precip_sensitivity{ov.weight_suffix(weighting)}"
    settings = {**SETTINGS, **run.settings(), "cell_weighting": weighting}

    if opts.plot_only:
        print(f"  -> {plot_comparison(out_dir, find_span(out_dir, weighting), weighting)}")
        return

    print("=" * 72)
    print("ERA5 per-cell cloud hours: effect of the no-precipitation filter")
    print(run.report())
    print("=" * 72)
    print(f"  pass 1 of 2: as is (tp < {SETTINGS['precip_rate_max']:g} mm hr-1)")
    C_as_is = count_cell_hours(build_args(settings))
    print("  pass 2 of 2: precipitation allowed")
    C_allowed = count_cell_hours(build_args({**settings, **PRECIP_ALLOWED}))

    span = ov.span_tag(C_as_is["seasons"])
    check_pair(C_as_is, C_allowed)

    as_is, allowed = hours_per_cell(C_as_is), hours_per_cell(C_allowed)
    check_against_existing_csvs(as_is, span, hour_count_dir, weighting)

    raw_as_is, raw_allowed = cell_hours(C_as_is), cell_hours(C_allowed)
    lost_hours = {c: percent_lost(as_is[c], allowed[c]) for c in PHASE_CATS}
    lost_cell_hours = {c: percent_lost(raw_as_is[c], raw_allowed[c]) for c in PHASE_CATS}

    print_tables(as_is, allowed, lost_hours, lost_cell_hours)
    print_rankings(as_is, allowed)

    n_written = write_all(as_is, allowed, lost_hours, lost_cell_hours, out_dir, span,
                          weighting)
    readme = write_readme(out_dir, span, C_as_is["has_site"], weighting)
    figure = plot_comparison(out_dir, span, weighting)
    print(f"\n  -> {n_written} CSV files + {readme.name} + {figure.name} in {out_dir}")


if __name__ == "__main__":
    main()
