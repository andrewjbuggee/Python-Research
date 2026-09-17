"""Build genie_arm_seasonal_hours.txt from Genie's source files.

The seasonal observation table the ERA5 comparison reads
(``genie_arm_seasonal_hours.txt``, see :func:`load_observations` in
``plot_lwp_histogram_by_surface_class``) is assembled here from three files
Genie supplied, kept beside the code under ``genie_obs_source/``:

``liquid_containing_clouds_yearly_totals_ver2.csv``
    Liquid-containing (liquid + mixed-phase) cloud, NON-precipitating, as
    COUNTS of 30 s samples per season. Divide by 120 for hours.
``ice_only_clouds_yearly_totals_ver2.csv``
    Ice-only cloud (no liquid, no mixed phase), non-precipitating, same
    units. These two are the "ver2" correction of 2026-09-16: the first
    version had left mixed-phase counts in the ice-only category.
``liquid-containing-clouds-total-hours-per-season.csv``
    The original (2026-09-01) table in hours, with six categories. Only its
    PRECIPITATING liquid and ice, clear-sky and others rows are used; its
    ``With Liquid`` and ``Ice Only`` rows are superseded by the ver2 files.

Why the precipitating rows can be carried over: for every season, the ver2
liquid + ice sum equals the original ``With Liquid + Ice Only`` sum to
within rounding (asserted below). The correction moved mixed-phase hours
between the two non-precipitating categories and changed nothing else, so
the precipitating, clear-sky and others rows -- and the derived missing
hours -- are unaffected. That is an inference from the numbers, not a
statement from Genie; the precipitating categories were not re-supplied.

``missing`` is derived, exactly as before: the season window (Oct 1 - Mar 31,
182 days or 183 when the ending year is a leap year) less the six supplied
categories.

Run from this directory::

    python build_genie_seasonal_hours.py

It overwrites ``genie_arm_seasonal_hours.txt``.
"""

from __future__ import annotations

import calendar
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "genie_obs_source"
OUT = HERE / "genie_arm_seasonal_hours.txt"

# Genie's ARM data are 30 s samples, so 120 samples make one hour.
SAMPLES_PER_HOUR = 120.0

# How closely the ver2 (liquid + ice) sum must reproduce the original
# (With Liquid + Ice Only) sum, per season, for the carry-over of the other
# rows to be justified. The original table is in whole hours, so 0.5 h of
# rounding per row -> 1 h on a two-row sum.
SUM_TOL_H = 1.0


def read_counts_csv(path: Path) -> np.ndarray:
    """A ver2 file: header ``,0`` then ``<index>,<count>`` per season."""
    rows = [r for r in csv.reader(path.open(encoding="utf-8-sig")) if r]
    assert rows[0] == ["", "0"], f"{path}: unexpected header {rows[0]!r}"
    idx = [int(r[0]) for r in rows[1:]]
    assert idx == list(range(len(idx))), f"{path}: seasons not 0..n-1"
    return np.array([float(r[1]) for r in rows[1:]])


def read_original_csv(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    """The original table: seasons across, categories down, in hours."""
    rows = [r for r in csv.reader(path.open(encoding="utf-8-sig")) if r]
    header = rows[0]
    seasons = [int(h.split("/")[0]) for h in header[1:]]     # '2014/15' -> 2014
    table = {}
    for r in rows[1:]:
        if not r[0]:
            continue                                        # blank spacer row
        table[r[0]] = np.array([float(v) for v in r[1:]])
    return seasons, table


def season_window_hours(seasons) -> np.ndarray:
    """Oct 1 - Mar 31 in hours: 182 days, 183 when the ending year is leap."""
    return np.array([(183.0 if calendar.isleap(y + 1) else 182.0) * 24.0
                     for y in seasons])


def main() -> None:
    seasons, orig = read_original_csv(
        SRC / "liquid-containing-clouds-total-hours-per-season.csv")
    liq_h = read_counts_csv(
        SRC / "liquid_containing_clouds_yearly_totals_ver2.csv") / SAMPLES_PER_HOUR
    ice_h = read_counts_csv(
        SRC / "ice_only_clouds_yearly_totals_ver2.csv") / SAMPLES_PER_HOUR
    n = len(seasons)
    assert liq_h.size == n and ice_h.size == n, "season counts differ"

    # The justification for carrying the other rows over -- see the module
    # docstring. Fails loudly if a future file breaks the pattern.
    old_sum = orig["With Liquid"] + orig["Ice Only"]
    new_sum = liq_h + ice_h
    gap = np.abs(new_sum - old_sum)
    assert gap.max() <= SUM_TOL_H, (
        f"ver2 liquid + ice differs from the original With Liquid + Ice Only "
        f"by up to {gap.max():.2f} h; the precipitating rows cannot be "
        f"assumed unchanged")

    liq_precip = orig["With Liquid (with precip)"]
    ice_precip = orig["Ice Only (with precip)"]
    clear = orig["Clear Sky"]
    others = orig["Others"]
    missing = (season_window_hours(seasons)
               - (liq_h + ice_h + liq_precip + ice_precip + clear + others))

    header = f"""\
# DOE ARM Utqiagvik seasonal cloud hours -- GENIE'S EXACT NUMBERS, ver2
# ============================================================================
#
# GENERATED by build_genie_seasonal_hours.py from the files under
# genie_obs_source/ -- edit those and re-run, do not edit this file.
#
# Sources, all supplied by Genie Lorenzo Pearson, under
#   ~/Documents/Scripps - UCSD/Ocean Visions/Genie Lorenzo Pearson/
#
#   with_liquid  liquid_containing_clouds_yearly_totals_ver2.csv  (2026-09-16)
#   ice_only     ice_only_clouds_yearly_totals_ver2.csv           (2026-09-16)
#   liq_precip, ice_precip, clear_sky, others
#                liquid-containing-clouds-total-hours-per-season.csv (2026-09-01)
#
# The ver2 files are COUNTS of 30 s samples; they are divided by 120 here.
# They correct the 2026-09-01 table, which had left mixed-phase counts in
# the ice-only category. In her words: "with liquid counts (including
# liquid and mixed phase clouds)"; "ice counts (including only ice, no
# liquid or mixed phase)". Both are NON-precipitating categories: for every
# season, ver2 liquid + ice equals the original With Liquid + Ice Only to
# within {SUM_TOL_H:g} h (checked on build), so the correction moved hours
# between those two rows and touched nothing else. The precipitating,
# clear-sky and others rows are therefore carried over from the original
# table unchanged. That is inferred from the numbers -- the precipitating
# categories were not re-supplied.
#
# Record means (all {n} seasons):
#   with_liquid {liq_h.mean():.1f} | ice_only {ice_h.mean():.1f}
#   liq_precip {liq_precip.mean():.1f} | ice_precip {ice_precip.mean():.1f}
#   clear_sky {clear.mean():.1f} | others {others.mean():.1f}
#   missing {missing.mean():.1f}
#
# SEASON LABELLING
# The original CSV's columns are headed '2014/15' .. '2024/25'; the ver2
# files index seasons 0..{n - 1} in the same order. This file uses the START
# year to match the ERA5 side, so '2014/15' is written as 2014.
#
# THE 'missing' COLUMN IS DERIVED, NOT SUPPLIED
#     missing = season window - (the six supplied categories)
# with a LEAP-AWARE window: Oct 1 - Mar 31 is 182 days, or 183 when the
# ending year is a leap year (2015/16, 2019/20, 2023/24). It only drives the
# 'incomplete season' flag ({100 * 0.05:g}% of the window), never a plotted
# quantity. The incomplete seasons are unchanged by ver2: 2016/17, 2019/20
# and 2020/21.
#
# Hours are per season for the single ARM site.
# ---------------------------------------------------------------------------
# season  with_liquid  ice_only  liq_precip  ice_precip  clear_sky  others  missing
"""
    lines = [header]
    for i, y in enumerate(seasons):
        lines.append(f"{y:<8}{liq_h[i]:>12.2f}{ice_h[i]:>10.2f}"
                     f"{liq_precip[i]:>12.0f}{ice_precip[i]:>12.0f}"
                     f"{clear[i]:>11.0f}{others[i]:>8.0f}{missing[i]:>9.2f}\n")
    OUT.write_text("".join(lines))
    print(f"wrote {OUT}  ({n} seasons)")
    print(f"  with_liquid mean {liq_h.mean():.1f} h   ice_only mean "
          f"{ice_h.mean():.1f} h   max |ver2 - original| sum gap "
          f"{gap.max():.2f} h")


if __name__ == "__main__":
    main()
